import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('172.30.0.1', 12345))
print("[INFO] Connected to the test server.")

# Receive size prefix
frame_size_data = client.recv(4)
frame_size = int.from_bytes(frame_size_data, 'big')
print(f"[DEBUG] Received frame size: {frame_size}")

# Receive data
data = client.recv(frame_size)
print(f"[DEBUG] Received data: {data.decode()}")
client.close()
