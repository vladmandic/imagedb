# ImageDB

Experiments in image search using two different datastores...

- **Index**  
  - Creates l2-normalized feature embedding for each image  
  - Extracts exif metadata and file stat  
  - Stores everything in database  
- **Search** 
  - Results are returned by similarity  
  - Search-by-image: create feature embedding and compare it for similarity with image vectors in database  
  - Search-by-prompt: encode prompt embedding and compare it for similarity with image vectors in database  
  - Search-by-text: search for text in metadata  

## Install

Includes shared requirements for both solutions  

```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## ImageDB-LanceDB

> python imagedb-lancedb.py --help  
> python imagedb-lance.py --index /home/vlado/dev/images/ --force  
> python imagedb-lance.py --search /home/vlado/dev/images/Siamese_100.jpg  

*Note*: More feature-rich than Faiss due to presence of unified store and hybrid search as well as conditional filters  
*Dependencies*:
- [OpenClip](https://github.com/mlfoundations/open_clip) for embedding extraction  
- [LanceDB](https://lancedb.github.io/lancedb/) as vector database with hybrid vector and metadata store and search  

## ImageDB-Faiss

> python imagedb-faiss.py --help  
> python imagedb-faiss.py --index /home/vlado/dev/images/  
> python imagedb-faiss.py --search /home/vlado/dev/images/Siamese_100.jpg  

*Note*: Uses separate vector and metadata stores which are searched separately  
*Dependencies*:
- [OpenClip](https://github.com/mlfoundations/open_clip) for embedding extraction  
- [Faiss](https://github.com/facebookresearch/faiss) as vector database  
- [Pandas](https://pandas.pydata.org/) for metadata store and search  

## Notes

### Performance

- **LanceDB**: *load=3.3sec, index=7384images/138sec its=52.5, search-by-100-images=2.0sec*
- **Faiss**: *load=3.0sec index=7384images/119sec its=62.0, search-by-100-images=1.7sec*

### Embedding Models

- OpenCLiP is chosen as its one of the few models families that can do both text and image feature extraction  
- Any CLiP model can be used  
  *Defaults*: Faiss=`ViT-L-14`, LanceDB=`ViT-B-16`  
- Most CliP models also have different variants depending on which dataset they were trained on  
  *Defaults*: Faiss=`laion2b_s32b_b82k`, LanceDB=`laion2b_s34b_b88k`  
