import subprocess

# Replace <username> and <ip_address> with the actual username and IP address of your remote server
ping_output = subprocess.Popen(['ping', '-c', '4', '-W', '1', '192.168.95.221'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True).communicate()

# Extract the average round-trip time (RTT) from the ping output
rtt = float(ping_output[0].split('rtt min/avg/max/mdev = ')[1].split('/')[1])

print(f"Latency to remote server: {rtt} ms")
