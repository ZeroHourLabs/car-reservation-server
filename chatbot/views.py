from rest_framework.decorators import api_view
from rest_framework.response import Response
from openai import OpenAI
from django.conf import settings

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=settings.OPENAI_API_KEY)

@api_view(['POST'])
def chat_with_openai(request):
    """OpenAI API와 연동된 챗봇 API"""
    user_message = request.data.get("message", "")

    if not user_message:
        return Response({"error": "메시지가 비어 있습니다."}, status=400)

    try:
        # OpenAI API 호출 (최신 방식)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_message}],
            max_tokens=150,
        )

        # 응답 메시지 추출
        reply = response.choices[0].message.content
        return Response({"reply": reply})

    except Exception as e:
        return Response({"error": f"서버 오류: {str(e)}"}, status=500)
