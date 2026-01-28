# ====================================================
# Pathogen/env/bacterial_evolution_env.py
# Biologically-Grounded Evolution Environment
# With Configurable Biological Parameters
# FIXED: Proper gene-sample integration
# ====================================================
# Quick fix for utils import
import sys
import os
if 'utils' not in sys.modules:
    utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
    sys.path.insert(0, os.path.dirname(utils_path))
    
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


class BacterialEvolutionEnv(gym.Env):
    """
    Bacterial Evolution Environment with:
    - Biologically-grounded rewards
    - Multi-relational graph structure
    - Gene-level state tracking
    - Configurable biological parameters for ablation studies
    - Fixed: Proper sampling of gene-rich isolates
    """

    def __init__(
        self, 
        data_path, 
        amr_path, 
        sample_size=100, 
        max_steps=20,
        # Biological parameters (configurable for ablation)
        antibiotic_pressure=0.5,          # Selection pressure [0=none, 1=lethal]
        mutation_gain_rate=0.25,          # Probability of gaining gene via mutation
        mutation_loss_rate=0.10,          # Probability of losing gene via mutation
        transfer_success_rate=0.40,       # Probability of successful HGT
        transfer_gene_range=(2, 5),       # Min/max genes transferred
        gene_similarity_threshold=0.3,    # Threshold for graph edges
        sparse_column_threshold=0.8       # Keep columns with >=80% non-null
    ):
        super(BacterialEvolutionEnv, self).__init__()
        
        # Store configuration
        self.data_path = data_path
        self.amr_path = amr_path
        self.max_steps = max_steps
        
        # Biological parameters
        self.antibiotic_pressure = antibiotic_pressure
        self.mutation_gain_rate = mutation_gain_rate
        self.mutation_loss_rate = mutation_loss_rate
        self.transfer_success_rate = transfer_success_rate
        self.transfer_gene_range = transfer_gene_range
        self.gene_similarity_threshold = gene_similarity_threshold
        self.sparse_column_threshold = sparse_column_threshold
        
        print("\n" + "="*70)
        print("Initializing Bacterial Evolution Environment")
        print("="*70)
        print(f"Configuration:")
        print(f" Antibiotic Pressure: {antibiotic_pressure:.2f}")
        print(f" Mutation Gain Rate: {mutation_gain_rate:.2%}")
        print(f" Mutation Loss Rate: {mutation_loss_rate:.2%}")
        print(f" Transfer Success Rate: {transfer_success_rate:.2%}")
        print(f" Gene Similarity Threshold: {gene_similarity_threshold:.2f}")
        print("="*70)

        # ----------------------------------------------------
        # 1. Load and preprocess data - FIXED VERSION
        # -----------------------------------------------------
        print("\n[1/6] Loading dataset...")
        self.data = pd.read_csv(data_path, low_memory=False)
        print(f" Loaded {len(self.data)} rows, {self.data.shape[1]} columns")

        # --- FIXED: Filter to samples with AMR genes ---
        if 'gene' in self.data.columns:
            # Keep only rows where gene is not null
            data_with_genes = self.data[self.data['gene'].notna()].copy()
            print(f" Filtered to {len(data_with_genes)} rows with AMR genes")
            
            if len(data_with_genes) > 0:
                self.data = data_with_genes
                print(f" Using gene-rich subset for training")
            else:
                print(" No genes in merged data - will load from AMR file separately")
        else:
            print(" No 'gene' column in merged data - will load from AMR file")

        # --- FIXED: Sample at isolate level, not row level ---
        if len(self.data) > sample_size and 'sample' in self.data.columns:
            # Get unique bacterial isolates
            unique_samples = self.data['sample'].unique()
            print(f" Found {len(unique_samples)} unique bacterial isolates with genes")
            
            # Sample N unique isolates
            n_samples_to_use = min(sample_size, len(unique_samples))
            selected_samples = np.random.choice(
                unique_samples, 
                size=n_samples_to_use, 
                replace=False
            )
            self.data = self.data[self.data['sample'].isin(selected_samples)].reset_index(drop=True)
            print(f" Sampled {n_samples_to_use} unique bacterial isolates")
            print(f" Total rows for these isolates: {len(self.data)}")
        elif len(self.data) > sample_size:
            self.data = self.data.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f" Sampled {len(self.data)} rows")

        # Drop sparse columns
        min_valid_rows = int(self.sparse_column_threshold * len(self.data))
        self.data = self.data.dropna(axis=1, thresh=min_valid_rows)
        print(f" Removed columns with >{100*(1-self.sparse_column_threshold):.0f}% missing data")

        # -----------------------------------------------------
        # 2. Identify column types
        # -----------------------------------------------------
        print("\n[2/6] Identifying column types...")
        possible_categorical = ['country', 'serovar', 'predicted_mobility']
        categorical_cols = [c for c in possible_categorical if c in self.data.columns]
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        mixed_cols = [
            c for c in self.data.columns
            if c not in categorical_cols + numeric_cols + ['resistance', 'sample', 'gene', 'product']
        ]
        
        print(f" Categorical: {len(categorical_cols)} {categorical_cols}")
        print(f" Numeric: {len(numeric_cols)}")
        print(f" Mixed: {len(mixed_cols)}")

        # ----------------------------------
        # 3. Encode and normalize
        # ----------------------------------
        print("\n[3/6] Encoding features...")
        encoded = self.data.copy()
        
        # Store original sample IDs for gene mapping
        self.sample_ids = encoded['sample'].unique() if 'sample' in encoded.columns else []
        
        le = LabelEncoder()

        for col in categorical_cols + mixed_cols:
            try:
                encoded[col] = le.fit_transform(encoded[col].astype(str))
            except Exception as e:
                print(f" Skipping {col}: {e}")
                encoded[col] = 0

        # Normalize numeric
        scaler = MinMaxScaler()
        if len(numeric_cols) > 0:
            try:
                encoded[numeric_cols] = scaler.fit_transform(encoded[numeric_cols])
            except Exception as e:
                print(f" Skipping normalization: {e}")

        # -------------------------------------------------
        # 4. Build feature matrix (one row per isolate)
        # -------------------------------------------------
        print("\n[4/6] Building feature matrix...")
        feature_cols = categorical_cols + mixed_cols + numeric_cols
        feature_cols = [c for c in feature_cols if c in encoded.columns]
        
        # Aggregate to one row per sample (take mean of numeric, mode of categorical)
        if 'sample' in encoded.columns:
            sample_features = encoded.groupby('sample')[feature_cols].mean().reset_index(drop=True)
            self.features = sample_features.fillna(0)
            print(f" Feature matrix: {self.features.shape} (one row per isolate)")
        else:
            self.features = encoded[feature_cols].fillna(0)
            print(f" Feature matrix: {self.features.shape}")

        # -------------------------------------------------
        # 5. Extract gene information per sample - FIXED
        # -------------------------------------------------
        print("\n[5/6] Extracting AMR gene associations...")
        self.sample_to_genes = self._extract_sample_genes()
        
        if len(self.sample_to_genes) > 0:
            gene_counts = [len(genes) for genes in self.sample_to_genes.values()]
            avg_genes = np.mean(gene_counts)
            max_genes = np.max(gene_counts)
            samples_with_genes = sum(1 for g in gene_counts if g > 0)
            print(f" Mapped {len(self.sample_to_genes)} samples to genes")
            print(f" Samples with genes: {samples_with_genes}/{len(self.sample_to_genes)}")
            print(f" Average genes per sample: {avg_genes:.1f}")
            print(f" Max genes in a sample: {max_genes}")
        else:
            print("  No gene mappings created")

        # ----------------------------------------------
        # 6. Build biological graph
        # ----------------------------------------------
        print("\n[6/6] Constructing biological graph...")
        self.edge_index, self.edge_type = self._build_biological_graph()
        print(f" Graph: {self.features.shape[0]} nodes, {self.edge_index.shape[1]} edges")

        # ------------------------------------------------
        # 7. Initialize reward calculator
        # ------------------------------------------------
        from utils.reward_functions import BiologicalRewardCalculator
        self.reward_calculator = BiologicalRewardCalculator(amr_path, antibiotic_pressure)

        # ------------------------------------------------
        # 8. Define RL spaces
        # ------------------------------------------------
        self.action_space = spaces.Discrete(3)  # 0=mutate, 1=transfer, 2=stable
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.features.shape[1],), dtype=np.float32
        )

        # State tracking (initialized here, set in reset())
        self.state = None
        self.current_step = 0
        self.current_sample_idx = 0
        self.current_genes = []

        print("\n" + "="*70)
        print("Environment initialization complete!")
        print("="*70 + "\n")

    # ==============================
    # HELPER METHODS
    # ==============================

    def _extract_sample_genes(self):
        """
        Extract which AMR genes belong to each sample.
        FIXED: Properly handles both merged data and separate AMR file.
        Returns dict: {sample_idx: [gene1, gene2, ...]}
        """
        sample_genes = {}
        
        # Check if we have required columns in merged data
        has_sample = 'sample' in self.data.columns
        has_gene = 'gene' in self.data.columns
        
        if not has_sample:
            print(" No 'sample' column - cannot map genes")
            return {i: [] for i in range(len(self.features))}
        
        # Strategy 1: Extract from merged data
        if has_gene:
            for idx, sample_id in enumerate(self.sample_ids):
                if idx >= len(self.features):
                    break
                
                # Get all genes for this sample (filter out NaN)
                sample_data = self.data[self.data['sample'] == sample_id]
                genes = sample_data['gene'].dropna().unique().tolist()
                sample_genes[idx] = genes
            
            # Check if we got meaningful results
            avg_genes = np.mean([len(g) for g in sample_genes.values()]) if sample_genes else 0
            if avg_genes > 0.5:
                return sample_genes
            else:
                print(f" Low gene count in merged data (avg={avg_genes:.2f}) - trying AMR file...")
        
        # Strategy 2: Load from separate AMR file
        try:
            amr_data = pd.read_csv(self.amr_path)
            
            if 'sample' not in amr_data.columns or 'gene' not in amr_data.columns:
                print(" AMR file missing required columns")
                return {i: [] for i in range(len(self.features))}
            
            # Build mapping for our selected samples
            for idx, sample_id in enumerate(self.sample_ids):
                if idx >= len(self.features):
                    break
                
                genes = amr_data[amr_data['sample'] == sample_id]['gene'].dropna().unique().tolist()
                sample_genes[idx] = genes
            
            print(f" Loaded genes from separate AMR file")
            return sample_genes
            
        except Exception as e:
            print(f" Error loading AMR file: {e}")
            return {i: [] for i in range(len(self.features))}

    def _build_biological_graph(self):
        """
        Construct multi-relational graph based on:
        1. Gene similarity (Cosine similarity with configurable threshold)
        2. Plasmid compatibility (same replicon type)
        3. Geographic proximity (same country)
        4. Serovar relatedness (same serovar)
        
        Uses self.gene_similarity_threshold for edge creation.
        """
        n_samples = len(self.features)
        edge_list = []
        edge_types = []  # 0=gene, 1=plasmid, 2=geo, 3=serovar
        
        # --- Edge Type 0: Gene Similarity ---
        if len(self.sample_to_genes) > 0:
            # Build gene presence matrix
            all_genes = set()
            for genes in self.sample_to_genes.values():
                all_genes.update(genes)
            all_genes = sorted(list(all_genes))
            
            if len(all_genes) > 0:
                # Create binary matrix
                gene_matrix = np.zeros((n_samples, len(all_genes)))
                for idx, genes in self.sample_to_genes.items():
                    if idx < n_samples:
                        for gene in genes:
                            if gene in all_genes:
                                gene_col = all_genes.index(gene)
                                gene_matrix[idx, gene_col] = 1
                
                # Compute similarity with configurable threshold
                if gene_matrix.sum() > 0:
                    similarity = cosine_similarity(gene_matrix)
                    
                    for i in range(n_samples):
                        for j in range(i+1, n_samples):
                            if similarity[i, j] > self.gene_similarity_threshold:
                                edge_list.extend([[i, j], [j, i]])
                                edge_types.extend([0, 0])
        
        # --- Edge Type 1: Plasmid Compatibility ---
        rep_type_cols = [c for c in self.data.columns if 'rep_type' in c.lower()]
        if len(rep_type_cols) > 0 and 'sample' in self.data.columns:
            rep_col = rep_type_cols[0]
            rep_types = self.data.groupby('sample')[rep_col].first()
            
            for i in range(min(n_samples, len(self.sample_ids))):
                for j in range(i+1, min(n_samples, len(self.sample_ids))):
                    if i < len(self.sample_ids) and j < len(self.sample_ids):
                        sample_i = self.sample_ids[i]
                        sample_j = self.sample_ids[j]
                        
                        if sample_i in rep_types.index and sample_j in rep_types.index:
                            rep_i = str(rep_types[sample_i])
                            rep_j = str(rep_types[sample_j])
                            
                            if rep_i == rep_j and rep_i not in ['nan', 'None', '', '-']:
                                edge_list.extend([[i, j], [j, i]])
                                edge_types.extend([1, 1])
        
        # --- Edge Type 2: Geographic Proximity ---
        if 'country' in self.data.columns and 'sample' in self.data.columns:
            countries = self.data.groupby('sample')['country'].first()
            
            for i in range(min(n_samples, len(self.sample_ids))):
                for j in range(i+1, min(n_samples, len(self.sample_ids))):
                    if i < len(self.sample_ids) and j < len(self.sample_ids):
                        sample_i = self.sample_ids[i]
                        sample_j = self.sample_ids[j]
                        
                        if sample_i in countries.index and sample_j in countries.index:
                            if countries[sample_i] == countries[sample_j]:
                                edge_list.extend([[i, j], [j, i]])
                                edge_types.extend([2, 2])
        
        # --- Edge Type 3: Serovar Relatedness ---
        if 'serovar' in self.data.columns and 'sample' in self.data.columns:
            serovars = self.data.groupby('sample')['serovar'].first()
            
            for i in range(min(n_samples, len(self.sample_ids))):
                for j in range(i+1, min(n_samples, len(self.sample_ids))):
                    if i < len(self.sample_ids) and j < len(self.sample_ids):
                        sample_i = self.sample_ids[i]
                        sample_j = self.sample_ids[j]
                        
                        if sample_i in serovars.index and sample_j in serovars.index:
                            if serovars[sample_i] == serovars[sample_j]:
                                edge_list.extend([[i, j], [j, i]])
                                edge_types.extend([3, 3])
        
        # --- Fallback: Minimal connectivity ---
        if len(edge_list) == 0:
            print(" No biological edges found - creating ring graph")
            edge_list = [[i, (i+1) % n_samples] for i in range(n_samples)]
            edge_list += [[(i+1) % n_samples, i] for i in range(n_samples)]
            edge_types = [0] * len(edge_list)
        
        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        # Print statistics
        type_names = ['Gene Similarity', 'Plasmid Compat.', 'Geographic', 'Serovar']
        for t_idx, t_name in enumerate(type_names):
            count = (edge_type == t_idx).sum().item()
            if count > 0:
                print(f" {t_name}: {count} edges")
        
        return edge_index, edge_type

    # ==============================
    # RL INTERFACE
    # ==============================

    def reset(self, seed=None, options=None):
        """
        Reset environment to a random bacterial sample.
        This ensures the agent learns policies that generalize across diverse isolates.
        """
        super().reset(seed=seed)
        
        # Select random sample (simulates sampling random patient/isolate)
        self.current_sample_idx = np.random.randint(0, len(self.features))
        self.state = self.features.iloc[self.current_sample_idx].values.astype(np.float32)
        self.current_step = 0
        
        # Initialize gene set for this sample
        self.current_genes = self.sample_to_genes.get(self.current_sample_idx, []).copy()
        
        return self.state, {}

    def step(self, action):
        """
        Execute evolutionary action and return biologically-grounded reward.
        
        Actions:
        0 = Mutate (point mutations, spontaneous gene gain/loss)
        1 = Transfer (horizontal gene transfer via plasmid conjugation)
        2 = Stable (maintain current genotype, minimal drift)
        
        Uses configurable biological parameters for stochastic events.
        """
        self.current_step += 1
        
        # Store genes before action (for reward calculation)
        old_genes = self.current_genes.copy()
        
        # --- ACTION EFFECTS ---
        if action == 0:  # MUTATION
            # Modify state with mutation noise
            # bacterial mutation rates ~10^-6 to 10^-9 per bp per generation
            mutation_strength = np.random.uniform(0.05, 0.15)
            noise = np.random.normal(0, mutation_strength, size=self.state.shape)
            self.state = np.clip(self.state + noise, 0, 1)
            
            # Stochastic gene gain (de novo resistance acquisition)
            if np.random.rand() < self.mutation_gain_rate:
                available_genes = list(self.reward_calculator.gene_to_resistance.keys())
                if available_genes and len(available_genes) > 0:
                    new_gene = np.random.choice(available_genes)
                    if new_gene not in self.current_genes:
                        self.current_genes.append(new_gene)
            
            # Stochastic gene loss (genetic drift or deletion)
            if len(self.current_genes) > 0 and np.random.rand() < self.mutation_loss_rate:
                lost_gene = np.random.choice(self.current_genes)
                self.current_genes.remove(lost_gene)
        
        elif action == 1:  # PLASMID TRANSFER (Horizontal Gene Transfer)
            # Larger state change due to plasmid acquisition
            # HGT can transfer 10-100 kb of DNA containing multiple genes
            transfer_strength = np.random.uniform(0.1, 0.25)
            noise = np.random.normal(0, transfer_strength, size=self.state.shape)
            self.state = np.clip(self.state + noise, 0, 1)
            
            # Gain multiple genes in single HGT event
            # Conjugation success rates 10-50% (Frost et al., 2005)
            if np.random.rand() < self.transfer_success_rate:
                available_genes = list(self.reward_calculator.gene_to_resistance.keys())
                if len(available_genes) >= 2:
                    min_transfer, max_transfer = self.transfer_gene_range
                    n_transfer = min(
                        np.random.randint(min_transfer, max_transfer + 1), 
                        len(available_genes)
                    )
                    transferred = np.random.choice(
                        available_genes, 
                        size=n_transfer, 
                        replace=False
                    )
                    for gene in transferred:
                        if gene not in self.current_genes:
                            self.current_genes.append(gene)
        
        else:  # STABLE (action == 2)
            # Minimal environmental drift only
            # Models stable population with no major genetic events
            drift = np.random.normal(0, 0.01, size=self.state.shape)
            self.state = np.clip(self.state + drift, 0, 1)
        
        # --- COMPUTE BIOLOGICAL REWARD ---
        reward = self.reward_calculator.compute_reward(
            old_genes=old_genes,
            new_genes=self.current_genes,
            action=action,
            state_features=self.state
        )
        
        # Episode termination
        done = self.current_step >= self.max_steps
        
        # Calculate additional metrics
        survival_prob = self.reward_calculator.compute_survival_probability(self.current_genes)
        
        info = {
            "step": self.current_step,
            "action": action,
            "action_name": ["mutate", "transfer", "stable"][action],
            "reward": reward,
            "n_genes": len(self.current_genes),
            "survival_prob": survival_prob,
            "genes_gained": len(self.current_genes) - len(old_genes),
            "antibiotic_pressure": self.antibiotic_pressure
        }
        
        return self.state, reward, done, False, info

    def get_edge_index(self):
        """Return graph edge index for GNN."""
        return self.edge_index
    
    def get_edge_type(self):
        """Return graph edge types (0=gene, 1=plasmid, 2=geo, 3=serovar)."""
        return self.edge_type
    
    def get_config(self):
        """Return environment configuration for logging."""
        return {
            "antibiotic_pressure": self.antibiotic_pressure,
            "mutation_gain_rate": self.mutation_gain_rate,
            "mutation_loss_rate": self.mutation_loss_rate,
            "transfer_success_rate": self.transfer_success_rate,
            "gene_similarity_threshold": self.gene_similarity_threshold,
            "max_steps": self.max_steps,
            "n_samples": len(self.features),
            "n_edges": self.edge_index.shape[1]
        }