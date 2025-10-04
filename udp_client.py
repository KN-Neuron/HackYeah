import time
import socket

while True:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.settimeout(1.0)
    message = b'test'
    addr = ("127.0.0.1", 11111)
    
    start = time.time()
    client_socket.sendto(message, addr)
    try:
        data, server = client_socket.recvfrom(1024)
        end = time.time()
        elapsed = end - start
        print(f'{data} {elapsed}')
        time.sleep(1)
    except socket.timeout:
        print('REQUEST TIMED OUT')