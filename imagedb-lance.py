# https://lancedb.github.io/lancedb/

from typing import Union, TYPE_CHECKING
from functools import partial
from datetime import datetime
import os
import re
import time
import json
import logging
from PIL import Image
from tqdm import tqdm

import torch
import lancedb
import lancedb.embeddings
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry, OpenClipEmbeddings, EmbeddingFunctionConfig

if TYPE_CHECKING:
    from lancedb.db import LanceTable


log = logging.getLogger('__name__')
ImageDBModels = [{'id': 0, 'name': 'ViT-B-16', 'pretrained': 'laion2b_s34b_b88k'}]


class ImageDBResult():
    def __init__(self, uri, vector, size, mtime, width, height, meta, _distance:float=0, _relevance_score:float=0, _score:float=0, ts:float=0): # pylint: disable=unused-argument
        self.folder = os.path.dirname(uri)
        self.file = os.path.basename(uri)
        self.size = size
        self.mtime = datetime.fromtimestamp(mtime)
        self.distance = round(_distance, 3)
        self.relevance = round(_relevance_score, 3)
        self.score = round(_score, 3)
        self.exists = os.path.exists(uri)
        self.width = width
        self.height = height
        self.meta = meta

        @property
        def image(self):
            return Image.open(uri) if self.exists else None

    def __str__(self):
        s = f'folder="{self.folder}" file="{self.file}" size={self.size} mtime="{self.mtime}" exists={self.exists} image={self.width}x{self.height}'
        if self.distance > 0:
            s += f' distance={self.distance}'
        if self.relevance > 0:
            s += f' relevance={self.relevance}'
        if self.score > 0:
            s += f' score={self.score}'
        return s


class ImageDB:
    def __init__(self, folder:str, device:str='cpu', dtype:str='fp16', batch:int=16, overwrite:bool=False, config:dict=None, force:bool=False):
        lancedb.embeddings.open_clip.tqdm = partial(tqdm, disable=True)
        self.config: dict = config or ImageDBModels[0]
        self.device: str = device
        self.dtype: torch.dtype = torch.bfloat16 if dtype == 'bf16' else torch.float16 if dtype == 'fp16' else torch.float32
        self.batch: int = batch
        self.folder: str = folder
        self.force: bool = force
        self.dim: int = 512
        # self.reranker = LinearCombinationReranker(weight=0.7, fill=1, return_score='relevance')
        self.db = lancedb.connect(self.folder)
        registry: EmbeddingFunctionRegistry = EmbeddingFunctionRegistry.get_instance()
        self.model: OpenClipEmbeddings = registry.get('open-clip').create(
            name=self.config['name'],
            pretrained=self.config['pretrained'],
            device=self.device,
            batch_size=self.batch,
            normalize=True
        )
        self.dim = self.model.ndims()

        class ImageDBSchema(LanceModel):
            uri: str = self.model.SourceField()
            vector: Vector(self.dim) = self.model.VectorField()
            size: int
            mtime: float
            width: int
            height: int
            meta: str
            ts: float

        self.table: LanceTable = self.db.create_table( # pylint: disable=unexpected-keyword-arg
            name='images',
            schema=ImageDBSchema,
            mode='overwrite' if overwrite else 'create',
            exist_ok=True,
            embedding_functions=[EmbeddingFunctionConfig(source_column='uri', vector_column='vector', function=self.model)],
        )
        self.model._model.to(device=self.device, dtype=self.dtype)
        model_id = f'{self.config["name"]}/{self.config["pretrained"]}'
        log.debug(f'ImageDB: db={self.db} model="{model_id}" dim={self.dim} device={self.device} batch={self.batch} overwrite={overwrite} rows={self.table.count_rows()}')
        try:
            table_metadata = json.loads(list(self.table.schema.metadata.values())[0].decode('utf-8'))[0]
            log.debug(f'ImageDB: metadata={table_metadata}')
        except Exception:
            pass

    def utf_decode(self, s: bytes): # decode byte-encoded exif metadata
        remove_prefix = lambda text, prefix: text[len(prefix):] if text.startswith(prefix) else text # pylint: disable=unnecessary-lambda-assignment
        for encoding in ['utf-8', 'utf-16', 'ascii', 'latin_1', 'cp1252', 'cp437']: # try different encodings
            try:
                s = remove_prefix(s, b'UNICODE')
                s = remove_prefix(s, b'ASCII')
                s = remove_prefix(s, b'\x00')
                val = s.decode(encoding, errors="strict")
                val = re.sub(r'[\x00-\x09\n\s\s+]', '', val).strip() # remove remaining special characters, new line breaks, and double empty spaces
                return val
            except Exception:
                pass
        return ''

    def get_metadata(self, image: Image.Image):
        try:
            exif = image._getexif() # pylint: disable=protected-access
        except Exception:
            exif = None
        if exif is None:
            return ''
        for k, v in exif.items():
            if k == 37510: # comment
                return self.utf_decode(v)
        return ''

    def enum_folder(self, folder: str, recursive: bool=True):
        batch_num = 0
        entries = []
        log.debug(f'ImageDB analyze: folder="{folder}"')
        for f in os.scandir(folder):
            if f.is_dir() and recursive:
                yield from self.enum_folder(f.path)
            if f.is_file() and os.path.splitext(f.name)[1].lower() in ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'):
                batch_num += 1
                stat = os.stat(f.path)
                image = Image.open(f.path)
                entries.append({
                    'uri': f.path,
                    'mtime': stat.st_mtime,
                    'size': stat.st_size,
                    'width': image.width,
                    'height': image.height,
                    'meta': self.get_metadata(image),
                    'ts': time.time(),
                })
                image.close()
            if batch_num >= self.batch:
                yield entries
                entries = []
                batch_num = 0
        yield entries

    def add_folder(self, folder: str, recursive: bool=True, index: bool=True):
        log.debug(f'ImageDB add: folder={folder} recursive={recursive} index={index} force={self.force}')
        if self.force:
            self.table.add(self.enum_folder(folder, recursive=recursive), on_bad_vectors='drop') # add using batch iterator
        else:
            self.table.merge_insert('"?table?".uri') \
                .when_matched_update_all() \
                .when_not_matched_insert_all() \
                .execute(self.enum_folder(folder, recursive=recursive), on_bad_vectors='drop') # sort-of-upsert using batch iterator
        if index:
            self.index()

    def search(self, query: Union[Image.Image, str], condition: str=None, limit: int=10, distance: float=2, score:float=0.01, mode: str='hybrid'):
        if os.path.isfile(query):
            query = Image.open(query)
            mode='auto'
        query_builder = self.table.search(query, query_type=mode)
        if condition and len(condition) > 0:
            query_builder = query_builder.where(condition)
        query_results = query_builder.limit(limit).to_list()
        filtered_results = [res for res in query_results if (res['_distance'] <= distance if '_distance' in res else True) and (res['_relevance_score'] > score if '_relevance_score' in res else True) and (res['_score'] > score if '_score' in res else True)]
        return [ImageDBResult(**res) for res in filtered_results]

    def index(self):
        try:
            self.table.create_index(vector_column_name="vector", num_sub_vectors=self.dim//8, replace=False)
        except Exception:
            pass # may fail if number of entries is too small
        try:
            self.table.create_fts_index('uri', use_tantivy=False, replace=False) # pylint: disable=unexpected-keyword-arg
            log.debug('ImageDB index create')
            self.table.create_fts_index('meta', use_tantivy=False, replace=False) # pylint: disable=unexpected-keyword-arg
            self.table.create_scalar_index('size', replace=False)
            self.table.create_scalar_index('mtime', replace=False)
            self.table.create_scalar_index('width', replace=False)
            self.table.create_scalar_index('height', replace=False)
        except Exception:
            log.debug('ImageDB index update')
            self.table.optimize() # if already exists, optimize to add new records, faster than reindexing

    @property
    def count(self):
        return self.table.count_rows()

    @classmethod
    def list_models(cls):
        import open_clip
        ImageDBModels.clear()
        ImageDBModels.append({'id': 0, 'name': 'ViT-B-16', 'pretrained': 'laion2B-s34B-b88K'})
        pretrained_models = open_clip.list_pretrained()
        for i in range(len(pretrained_models)):
            ImageDBModels.append({'id': 100+i, 'name': pretrained_models[i][0], 'pretrained': pretrained_models[i][1]})
        return ImageDBModels


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(levelname)s: %(message)s')
    log.setLevel(logging.DEBUG)
    log.info('ImageDB start')
    parser = argparse.ArgumentParser(description = 'imagedb')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--search', action='store_true', help='run search')
    group.add_argument('--index', action='store_true', help='run indexing')
    group.add_argument('--list', action='store_true', help='list available models')
    parser.add_argument('--db', type=str, default='lancedb', help='database folder path')
    parser.add_argument('--id', type=int, default=0, help='model id')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'ipu', 'xpu', 'mkldnn', 'opengl', 'opencl', 'ideep', 'hip', 've', 'fpga', 'maia', 'xla', 'lazy', 'vulkan', 'mps', 'meta', 'hpu', 'mtia', 'privateuseone'], help='device to use')
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16', 'fp32'], help='force fp32 precision')
    parser.add_argument('--overwrite', action='store_true', help='index: overwrite database')
    parser.add_argument('--force', action='store_true', help='index: force add to database without duplicate check')
    parser.add_argument('--recursive', action='store_true', help='index: recursive folder indexing')
    parser.add_argument('--condition', type=str, default='', help='search: additional search condition, e.g. "width>1024"')
    parser.add_argument('--limit', type=int, default=20, help='search: limit number of results')
    parser.add_argument('--repeat', type=int, default=1, help='search: repeat search n times for benchmark')
    parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'hybrid', 'fts', 'vector'], help='search: mode')
    parser.add_argument('--cache', default='/mnt/models/huggingface', help='model cache folder')
    parser.add_argument('input', nargs='*', default='', help='search: prompt describing the image, text in metadata or path to image file; index: path to folder(s) containing images to index')
    args = parser.parse_args()

    os.environ.setdefault('HF_HUB_CACHE', args.cache)

    db = None
    if args.list:
        ImageDB.list_models()
        for model in ImageDBModels:
            log.info(model)
    else:
        if args.id > 0:
            ImageDB.list_models()
        db = ImageDB(
            folder=args.db,
            device=args.device,
            dtype=args.dtype,
            batch=16,
            overwrite=args.overwrite,
            force=args.force,
            config=ImageDBModels[args.id],
        )
    if not db:
        exit(0)

    if args.index:
        for input_folder in args.input:
            if not input_folder or len(input_folder) == 0 or not os.path.exists(input_folder):
                continue
            starting_n = db.count
            t0 = time.time()
            log.info(f'Index start: folder="{input_folder}"')
            db.add_folder(
                folder=input_folder,
                recursive=args.recursive,
                index=True,
            )
            t1 = time.time()
            log.info(f'Index end: images={db.count-starting_n} total={db.count} time={t1-t0:.3f} its={(db.count-starting_n)/(t1-t0):.2f}')

    if args.search:
        for input_query in args.input:
            if not input_query or len(input_query) == 0:
                continue
            t0 = time.time()
            log.info(f'Search start: {input_query}')
            for _n in range(args.repeat):
                items = db.search(
                    query=input_query,
                    condition=args.condition,
                    limit=args.limit,
                    mode=args.mode,
                )
            t1 = time.time()
            for item in items:
                log.info(item)
            log.info(f'Search end: time={t1-t0:.3f}')
