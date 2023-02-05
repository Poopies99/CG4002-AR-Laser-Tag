import socket

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Get server hostname and port
host = "192.168.95.221"
port = 12345

# Connect to server
client_socket.connect((host, port))

while True:
    message = input('Enter message to send to the server: ')
    if message == 'q':
        break

    # Send message to server
    client_socket.sendall(message.encode())

# Close the client socket
client_socket.close()
