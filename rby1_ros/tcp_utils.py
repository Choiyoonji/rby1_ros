
import socket
import pickle
import struct

# === TCP 통신 유틸 (inference.py와 동일 프로토콜) ===
def send_packet(sock, data_obj):
    """Length-prefixed pickle send/recv."""
    data_bytes = pickle.dumps(data_obj)
    sock.sendall(struct.pack('>I', len(data_bytes)) + data_bytes)

    data_len_bytes = sock.recv(4)
    if not data_len_bytes:
        raise ConnectionAbortedError("Server closed connection while waiting for message length.")
    msg_len = struct.unpack('>I', data_len_bytes)[0]

    response_data = b""
    while len(response_data) < msg_len:
        packet = sock.recv(msg_len - len(response_data))
        if not packet:
            raise ConnectionAbortedError("Server closed connection while receiving message.")
        response_data += packet
    return pickle.loads(response_data)

def connect_tcp(server_ip, server_port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((server_ip, server_port))
        sock.settimeout(None)  # recv timeout (초)
        return sock, True
    except Exception as e:
        return None, False
    
def disconnect_tcp(sock):
    if sock is not None:
        try:
            sock.close()
        except Exception:
            pass