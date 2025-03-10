from rest_framework.decorators import api_view
from rest_framework.response import Response
import chromadb
import requests
import time
import hashlib
import os

# ChromaDB 클라이언트 초기화
chroma_client = chromadb.PersistentClient(path='./chroma_db')
# collection 초기화
collection = chroma_client.get_collection(name="pdf_collection")

# 임베딩 캐시를 위한 딕셔너리
embedding_cache = {}


# 임베딩을 요청하는 함수 (캐시 적용)
def get_embedding(user_message):
    # 메시지 해시값을 사용해 캐시 확인
    message_hash = hashlib.md5(user_message.encode()).hexdigest()

    if message_hash in embedding_cache:
        print("캐시된 임베딩 사용")
        return embedding_cache[message_hash]

    try:
        url = "http://127.0.0.1:1234/v1/embeddings"  # 로컬 임베딩 모델 API
        payload = {
            "model": "text-embedding-intfloat-multilingual-e5-large-instruct",
            "input": user_message
        }
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, json=payload, headers=headers)
        time.sleep(1)  # 요청 간 딜레이 추가

        if response.status_code == 200:
            embedding = response.json()['data'][0]['embedding']
            print("Embedding: " + str(embedding))  # 리스트를 문자열로 변환하여 출력
            # 캐시 저장
            embedding_cache[message_hash] = embedding
            return embedding
        else:
            raise Exception(f"임베딩 생성 실패: {response.text}")
    except Exception as e:
        raise Exception(f"임베딩 생성 실패: {str(e)}")


# ChromaDB에서 임베딩을 검색하는 함수
def search_in_chromadb(user_message):
    query_embedding = get_embedding(user_message)
    results = collection.query(query_embedding, n_results=1)
    return results


# 이미 처리된 응답을 저장할 캐시
response_cache = {}


@api_view(['POST'])
def chat_with_local(request):
    user_message = request.data.get("message", "")
    if not user_message:
        return Response({"error": "메시지가 비어 있습니다."}, status=400)

    # 메시지에 대한 캐시된 응답이 있는지 확인
    if user_message in response_cache:
        print("캐시된 응답 반환")
        return Response({"reply": response_cache[user_message]})

    try:
        # ChromaDB에서 관련 문서 검색
        related_documents = search_in_chromadb(user_message)
        context = "\n".join(related_documents)

        # LM Studio 로컬 모델 호출
        url = "http://127.0.0.1:1234/v1/chat/completions"
        payload = {
            "model": "cohereforai.aya-expanse-8b",
            "messages": [
                {"role": "system", "content": "당신은 메르세데스-벤츠 차량 추천 전문가입니다. 사용자의 질문에 응답할 때 반드시 ChromaDB에 저장된 벤츠 차량 데이터만을 기반으로 답변해야 합니다. 데이터에 없는 정보는 '모르겠습니다.'라고 답변하세요."},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": context}
            ],
            "max_tokens": 8192
        }
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, json=payload, headers=headers)
        print("user_message: " + user_message)
        time.sleep(1)

        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
            # 응답 캐시 저장
            response_cache[user_message] = reply
            return Response({"reply": reply})
        else:
            return Response({"error": f"응답 실패: {response.text}"}, status=500)
    except Exception as e:
        return Response({"error": f"서버 오류: {str(e)}"}, status=500)
