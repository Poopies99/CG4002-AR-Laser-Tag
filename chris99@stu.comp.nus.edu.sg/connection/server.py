import socket

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Get the hostname and port
host = "0.0.0.0" # Listen on all available network interfaces
port = 12345

# Bind the socket to the host and port
server_socket.bind((host, port))

# Listen for incoming connections
server_socket.listen(5) # The number of queued connections

while True:
    # Accept a new client connection
    client_socket, client_address = server_socket.accept()

    print("Connection from:", client_address)

    # Receive data from the client
    data = client_socket.recv(1024)
    print("Received:", data)

    # Send the data back to the client
    client_socket.send(data)

    # Close the client socket
    client_socket.close()
