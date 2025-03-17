from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import NotebookLoader, UnstructuredFileLoader, TextLoader
import os, re
from loguru import logger
from langchain_text_splitters import Language

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
#                                    model_kwargs={'device': 'cpu'})
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
# embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # 384维


error_files = []


def process_document(doc, directory_path, ext):
    """处理单个文档的分块逻辑"""
    if isinstance(doc, tuple):
        page_content = doc[0]
        metadata = doc[1] if doc[1] is not None and len(doc[1]) else {'source': 'unknown'}  # 添加默认元数据

        doc = Document(page_content=page_content, metadata=metadata)
    file_path = doc.metadata['source']
    if not os.path.exists(file_path):
        error_files.append(file_path)
        return []

    category = os.path.relpath(file_path, directory_path).split(os.sep)[0]
    file_ext = ext

    # 配置Markdown标题分割器
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # 根据文件类型选择分割器
    if file_ext in ('.md', '.mdx'):
        # 清除jsx标签
        cleaned = re.sub(r'</?[a-zA-Z]+>', '', doc.page_content)
        split_content = markdown_splitter.split_text(cleaned)
    elif file_ext == '.ipynb':
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=1000,
            chunk_overlap=200
        )
        split_content = splitter.split_documents([doc])
    else:  # 处理txt等其他文本文件
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n\n', '\n\n', '\n'],
            chunk_size=1000,
            chunk_overlap=200
        )
        split_content = text_splitter.split_documents([doc])

    # 继承元数据
    for chunk in split_content:
        chunk.metadata.update({'category': file_ext, 'source': file_path})

    return split_content


def batch_import_guides(directory_path, persist_path=None, batch_size=10):
    """
    批量导入指南文档并生成向量库
    :param directory_path: 文档目录路径
    :param persist_path: 持久化存储路径（可选）
    """
    # 初始化加载器
    loaders = [
        # (DirectoryLoader(
        #     directory_path,
        #     glob="**/*.md",
        #     loader_cls=TextLoader,
        #     loader_kwargs={"autodetect_encoding": True},
        #     recursive=True,
        #     show_progress=True,
        #     silent_errors=True,
        #     use_multithreading=True
        # ), '.md'),
        (DirectoryLoader(
            directory_path,
            glob="**/*.mdx",
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            use_multithreading=True,
            recursive=True,
            show_progress=True,
            silent_errors=True,
        ), '.mdx'),
        # (DirectoryLoader(
        #     directory_path,
        #     glob="**/*.ipynb",
        #     loader_cls=NotebookLoader,
        #     loader_kwargs={"include_outputs": True,
        #                    "max_output_length": 500,
        #                    "remove_newline": True},
        #     recursive=True,
        #     show_progress=True,
        #     silent_errors=True,
        #     use_multithreading=True
        # ), '.ipynb')
    ]

    db = FAISS.from_texts(
        texts=[""],
        embedding=embeddings,
        metadatas=[{}]
    )  # 使用包含空字符串的初始化方式
    processed_count = 0

    for loader, ext in loaders:
        docs = loader.load()
        current_batch = []
        for doc in docs:
            # logger.warning(f'doc: {doc}')
            chunks = process_document(doc, directory_path, ext=ext)
            current_batch.extend(chunks)

        if current_batch:
            partial_db = FAISS.from_documents(current_batch, embeddings)
            db.merge_from(partial_db)
            processed_count += len(current_batch)

            # 达到批次大小时自动保存
            if persist_path and processed_count >= batch_size:
                db.save_local(persist_path)
                processed_count = 0

    # 最终保存剩余文档
    if persist_path and processed_count > 0:
        db.save_local(persist_path)

    return db


def load_faiss(persist_path, docs_dir):
    """加载本地存储的FAISS向量库"""

    # 初始化加载
    if not os.path.exists(persist_path):
        logger.warning(f'初始化加载')
        return batch_import_guides(docs_dir, persist_path)

    # 检查版本是否需要更新
    version_file = os.path.join(persist_path, "version.txt")
    current_version = str(os.path.getmtime(docs_dir))

    if (os.path.exists(version_file)
                      and open(version_file).read() != current_version):
        logger.warning(f'更新数据')
        return batch_import_guides(docs_dir, persist_path)

    # 直接加载
    if os.path.exists(os.path.join(persist_path, "index.faiss")):
        logger.warning(f'直接加载')
        return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)

