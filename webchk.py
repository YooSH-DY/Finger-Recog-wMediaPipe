import asyncio
import websockets
import json
from datetime import datetime


async def handle_client(websocket):
    """클라이언트 연결을 처리하는 함수"""
    client_address = websocket.remote_address
    print(f"✅ 새 클라이언트 연결: {client_address}")

    try:
        async for message in websocket:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            try:
                # JSON 데이터인지 확인
                data = json.loads(message)
                print(f"[{timestamp}] JSON 데이터 from {client_address}:")
                print(json.dumps(data, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                # 일반 텍스트 데이터
                print(f"[{timestamp}] 텍스트 데이터 from {client_address}: {message}")

            print("-" * 50)

    except websockets.exceptions.ConnectionClosed:
        print(f"❌ 클라이언트 연결 종료: {client_address}")
    except Exception as e:
        print(f"❌ 클라이언트 처리 중 오류 발생: {e}")


async def start_server():
    """웹소켓 서버 시작"""
    host = "0.0.0.0"  # 모든 네트워크 인터페이스에서 접근 허용
    port = 5678

    print(f"웹소켓 서버 시작 중... {host}:{port}")

    server = await websockets.serve(handle_client, host, port)
    print(f"✅ 웹소켓 서버가 {host}:{port}에서 실행 중입니다.")
    print("아이폰 앱에서 연결을 기다리는 중...")
    print("Ctrl+C를 눌러 서버를 종료할 수 있습니다.")

    await server.wait_closed()


def main():
    """메인 함수"""
    print("웹소켓 서버 시작")

    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")


if __name__ == "__main__":
    main()
