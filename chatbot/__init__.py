# chatbot/__init__.py
import os
import time

# 수정된 코드
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import chromadb
import requests
from django.conf import settings

# ChromaDB 클라이언트 초기화
chroma_client = chromadb.PersistentClient(path='./chroma_db')
# 컬렉션 이름
collection_name = "pdf_collection"

# 컬렉션이 이미 존재하는지 확인
if collection_name in chroma_client.list_collections():
    # 컬렉션이 존재하면 삭제
    chroma_client.delete_collection(collection_name)

# 새로 컬렉션을 생성
collection = chroma_client.create_collection(name=collection_name)

# LM Studio API로 로컬 임베딩 모델 호출
def get_embedding(user_message):
    try:
        # 로컬 모델에 요청 보내기
        url = "http://127.0.0.1:1234/v1/embeddings"  # 로컬 모델 API URL
        payload = {
            "model": "text-embedding-intfloat-multilingual-e5-large-instruct",
            "input": user_message
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",  # 필요시 API 키 포함
        }

        # POST 요청을 통해 임베딩을 요청
        response = requests.post(url, json=payload, headers=headers)
        print("user_message: "+user_message)
        time.sleep(1)

        # 응답에서 임베딩 벡터 추출
        if response.status_code == 200:
            response_data = response.json()
            embedding = response_data['data'][0]['embedding']  # 임베딩 벡터 추출
            print("Embedding: " + str(embedding))  # 리스트를 문자열로 변환하여 출력
            return embedding
        else:
            raise Exception(f"임베딩 생성 실패: {response.text}")

    except Exception as e:
        raise Exception(f"임베딩 생성 실패: {str(e)}")


def initialize_chromadb():
    """앱 초기화 시 PDF 파일을 읽어 임베딩을 ChromaDB에 업로드하는 함수"""
    pdf_directory = settings.PDF_URL  # PDF 파일이 저장된 디렉토리

    # PDF 디렉토리 내 모든 PDF 파일을 처리
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)

            # 이미 ChromaDB에 임베딩된 파일이 있으면 삭제
            existing_ids = collection.get(where={"source": filename})  # 기존 ID 확인
            if existing_ids["ids"]:  # 이미 임베딩된 파일이 있으면
                print(f"PDF '{filename}'는 이미 임베딩되어 있습니다. 삭제 후 다시 임베딩을 추가합니다.")
                collection.delete(where={"source": filename})  # 기존 임베딩 삭제

            # PyPDFLoader로 PDF 텍스트 로드
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            # CharacterTextSplitter로 텍스트 분리
            text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
            chunks = text_splitter.split_documents(documents)

            # 각 텍스트 덩어리별로 임베딩을 구하고 ChromaDB에 추가
            for chunk in chunks:
                embedding = get_embedding(chunk.page_content)  # 각 텍스트 덩어리 임베딩
                collection.add(
                    documents=[chunk.page_content],  # 텍스트 내용
                    embeddings=[embedding],  # 텍스트의 임베딩
                    metadatas=[{"source": filename}],  # 메타데이터 예: 파일 이름
                    ids=[f"{filename}_{chunk.metadata['page']}"]  # 고유한 ID 생성
                )

            print(f"PDF '{filename}' 임베딩 추가 완료.")



# 앱이 초기화될 때 해당 작업 실행
initialize_chromadb()
