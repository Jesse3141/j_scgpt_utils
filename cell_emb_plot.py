#export
#import
import json
import os
from pathlib import Path
from typing import Optional, Union
from sklearn.metrics import mutual_info_score

import numpy as np
import scanpy as sc
import pandas as pd
import torch
import short_utils

from anndata import AnnData
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm


from scgpt import logger
from scgpt.data_collator import DataCollator
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.utils import load_pretrained


PathLike = Union[str, os.PathLike]
#export
def get_batch_cell_embeddings(
    adata,
    cell_embedding_mode: str = "cls",
    model=None,
    vocab=None,
    max_length=1200,
    batch_size=64,
    model_configs=None,
    gene_ids=None,
    use_batch_labels=False,
) -> np.ndarray:
    """
    Get the cell embeddings for a batch of cells.

    Args:
        adata (AnnData): The AnnData object.
        cell_embedding_mode (str): The mode to get the cell embeddings. Defaults to "cls".
        model (TransformerModel, optional): The model. Defaults to None.
        vocab (GeneVocab, optional): The vocabulary. Defaults to None.
        max_length (int): The maximum length of the input sequence. Defaults to 1200.
        batch_size (int): The batch size for inference. Defaults to 64.
        model_configs (dict, optional): The model configurations. Defaults to None.
        gene_ids (np.ndarray, optional): The gene vocabulary ids. Defaults to None.
        use_batch_labels (bool): Whether to use batch labels. Defaults to False.

    Returns:
        np.ndarray: The cell embeddings.
    """

    count_matrix = adata.X
    count_matrix = (
        count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
    )
    print("loaded count matrix")

    # gene vocabulary ids
    if gene_ids is None:
        gene_ids = np.array(adata.var["id_in_vocab"])
        assert np.all(gene_ids >= 0)

    if use_batch_labels:
        batch_ids = np.array(adata.obs["batch_id"].tolist())

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, count_matrix, gene_ids, batch_ids=None):
            self.count_matrix = count_matrix
            self.gene_ids = gene_ids
            self.batch_ids = batch_ids

        def __len__(self):
            return len(self.count_matrix)

        def __getitem__(self, idx):
            row = self.count_matrix[idx]
            nonzero_idx = np.nonzero(row)[0]
            values = row[nonzero_idx]
            genes = self.gene_ids[nonzero_idx]
            # append <cls> token at the beginning
            genes = np.insert(genes, 0, vocab["<cls>"])
            values = np.insert(values, 0, model_configs["pad_value"])
            genes = torch.from_numpy(genes).long()
            values = torch.from_numpy(values)
            output = {
                "id": idx,
                "genes": genes,
                "expressions": values,
            }
            if self.batch_ids is not None:
                output["batch_labels"] = self.batch_ids[idx]
            return output

    if cell_embedding_mode == "cls":
        dataset = Dataset(
            count_matrix, gene_ids, batch_ids if use_batch_labels else None
        )
        print("created dataset")
        collator = DataCollator(
            do_padding=True,
            pad_token_id=vocab[model_configs["pad_token"]],
            pad_value=model_configs["pad_value"],
            do_mlm=False,
            do_binning=True,
            max_length=max_length,
            sampling=True,
            keep_first_n_tokens=1,
        )
        print("created collator")

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=collator,
            drop_last=False,
            num_workers=min(len(os.sched_getaffinity(0)), batch_size),
            pin_memory=True,
        )
        print("created data loader")

        device = next(model.parameters()).device
        print("created device")
        cell_embeddings = np.zeros(
            (len(dataset), model_configs["embsize"]), dtype=np.float32
        )
        print("created intial cell embeddings")
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            count = 0
            for data_dict in tqdm(data_loader, desc="Embedding cells"):
                input_gene_ids = data_dict["gene"].to(device)
                src_key_padding_mask = input_gene_ids.eq(
                    vocab[model_configs["pad_token"]]
                )
                print(" input gene ids to device")
                embeddings = model._encode(
                    input_gene_ids,
                    data_dict["expr"].to(device),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=data_dict["batch_labels"].to(device)
                    if use_batch_labels
                    else None,
                )
                print("encoded embeddings")

                embeddings = embeddings[:, 0, :]  # get the <cls> position embedding
                embeddings = embeddings.cpu().numpy()
                cell_embeddings[count : count + len(embeddings)] = embeddings
                count += len(embeddings)
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )
    else:
        raise ValueError(f"Unknown cell embedding mode: {cell_embedding_mode}")
    return cell_embeddings


#export
def embed_data(
    adata_or_file: Union[AnnData, PathLike],
    model_dir: PathLike,
    cell_type_key: str = "cell_type",
    gene_col: str = "feature_name",
    max_length=1200,
    batch_size=64,
    obs_to_save: Optional[list] = None,
    device: Union[str, torch.device] = "cuda",
    use_fast_transformer: bool = True,
    return_new_adata: bool = False,
) -> AnnData:
    """
    Preprocess anndata and embed the data using the model.

    Args:
        adata_or_file (Union[AnnData, PathLike]): The AnnData object or the path to the
            AnnData object.
        model_dir (PathLike): The path to the model directory.
        cell_type_key (str): The key in adata.obs that contains the cell type labels.
            Defaults to "cell_type".
        gene_col (str): The column in adata.var that contains the gene names.
        max_length (int): The maximum length of the input sequence. Defaults to 1200.
        batch_size (int): The batch size for inference. Defaults to 64.
        obs_to_save (Optional[list]): The list of obs columns to save in the output adata.
            If None, will only keep the column of :attr:`cell_type_key`. Defaults to None.
        device (Union[str, torch.device]): The device to use. Defaults to "cuda".
        use_fast_transformer (bool): Whether to use flash-attn. Defaults to True.
        return_new_adata (bool): Whether to return a new AnnData object. If False, will
            add the cell embeddings to a new :attr:`adata.obsm` with key "X_scGPT".

    Returns:
        AnnData: The AnnData object with the cell embeddings.
    """
    if isinstance(adata_or_file, AnnData):
        adata = adata_or_file
    else:
        adata = sc.read_h5ad(adata_or_file)

    # verify cell type key and gene col
    assert cell_type_key in adata.obs
    if gene_col == "index":
        adata.var["index"] = adata.var.index
    else:
        assert gene_col in adata.var

    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Using CPU instead.")

    # LOAD MODEL
    model_dir = Path(model_dir)
    vocab_file = model_dir / "vocab.json"
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]

    # vocabulary
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    adata.var["id_in_vocab"] = [
        vocab[gene] if gene in vocab else -1 for gene in adata.var[gene_col]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    # Binning will be applied after tokenization. A possible way to do is to use the unified way of binning in the data collator.

    vocab.set_default_index(vocab["<pad>"])
    genes = adata.var[gene_col].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)

    # all_counts = adata.layers["counts"]
    # num_of_non_zero_genes = [
    #     np.count_nonzero(all_counts[i]) for i in range(all_counts.shape[0])
    # ]
    # max_length = min(max_length, np.max(num_of_non_zero_genes) + 1)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_configs["dropout"],
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=False,
    )
    load_pretrained(model, torch.load(model_file), verbose=False)
    model.to(device)
    model.eval()
    print("loaded model")

    # get cell embeddings
    cell_embeddings = get_batch_cell_embeddings(
        adata,
        cell_embedding_mode="cls",
        model=model,
        vocab=vocab,
        max_length=max_length,
        batch_size=batch_size,
        model_configs=model_configs,
        gene_ids=gene_ids,
        use_batch_labels=False,
    )
    print("got cell embeddings")

    if return_new_adata:
        obs_to_save = [cell_type_key] if obs_to_save is None else obs_to_save
        obs_df = adata.obs[obs_to_save]
        return sc.AnnData(X=cell_embeddings, obs=obs_df, dtype="float32")

    adata.obsm["X_scGPT"] = cell_embeddings
    return adata
#export
base_dir = short_utils.get_base_dir()
base_dir

#export
# load data
full_adata = sc.read_h5ad(base_dir / 'training_data/tcga/genexp_data/xena_pan_can_genexp_clin.h5ad')
# load and examine the data at data/brca_scrna_epithelial.h5ad


#export
#truncate the data to 1000 cells
my_adata = full_adata[:10]
#export
# prep args for embed:

#if plot by label, set the cell type arg to the cool with label

embed_args = {'adata_or_file': my_adata,
              'model_dir': Path(base_dir / 'scgpt/models/scGPT_pancancer'),
              'cell_type_key': "tumor_type",
                'gene_col': "hgnc_gene",
              'max_length' : 2000,
              'batch_size' : 1,
              'obs_to_save':  None,
              'device':  "cuda",
              'use_fast_transformer': False,
              'return_new_adata':  True,
              }
#export
cell_embbed = embed_data(**embed_args)

#clean cell output
#export
#save the cell_embbed
cell_embbed.obs = my_adata.obs.copy()
#export
#plot the result cell_embbed.X which is #num cell rows of 512 collums
import umap
import matplotlib.pyplot as plt
#export
projection_data = my_adata.X
#export
#fit the projection
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(projection_data)
#export
# create a PCA

from sklearn.decomposition import PCA
import pandas as pd

# Perform PCA on the embeddings
pca = PCA(n_components=2)
pca_result = pca.fit_transform(projection_data)

# Create a DataFrame for the PCA results
pca_df = pd.DataFrame(pca_result, columns=['PCA 1', 'PCA 2'])
pca_df.index = my_adata.obs.index

# If you have the obs data as a DataFrame named my_adata.obs, concatenate it with the PCA results
full_df = pd.concat([my_adata.obs, pca_df], axis=1)

#export
import plotly.express as px

# Create a PCA plot
fig = px.scatter(pca_df, x='PCA 1', y='PCA 2',color=my_adata.obs['tumor_type'], title="PCA Plot")
fig.update_layout(xaxis_title="PCA 1", yaxis_title="PCA 2")
fig.show()
#export
#print explained variance ratio
print('explained variance by pc 1 & 2: ',pca.explained_variance_)
#export
import plotly.express as px

# Prepare your data
umap_x = embedding[:, 0]
umap_y = embedding[:, 1]


#export

# Create a DataFrame for Plotly: add the UMAP cols to my_adata.obs
umap_df = pd.DataFrame()
umap_df['UMAP 1'] = umap_x
umap_df['UMAP 2'] = umap_y

plt_title = 'UMAP all tcga erbb2 scgpt all genes '

#export
# Create a Plotly figure
fig = px.scatter(umap_df, x='UMAP 1', y='UMAP 2', color='tumor_type',
                 color_continuous_scale=['darkblue', 'red'],
                 title=plt_title,
                 labels={'Label': 'cancer'},
                 opacity=0.8)

# Update layout
fig.update_layout(legend_title_text='Cancer',
                  xaxis_title='UMAP 1',
                  yaxis_title='UMAP 2')

# Show the plot
fig.show()