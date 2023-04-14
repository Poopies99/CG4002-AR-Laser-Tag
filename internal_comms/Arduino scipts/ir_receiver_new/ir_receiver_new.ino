#pragma pack(1)

/*---------------- Data structures ----------------*/

enum PacketType
{
  HELLO,
  ACK,
  NACK,
  DATA
};

typedef struct
{
  uint8_t header;           // 1 byte header: 4 bit node id | 2 bit packet type | 2 bit sequence no
  uint8_t padding;          // padding header to 2 bytes
  int euler_x;              // contains IR data for data packet for IR sensors
  int euler_y;              // all other fields padded with 0 for data packet for IR sensors
  int euler_z;
  int acc_x;
  int acc_y;
  int acc_z;
  int flex_1;
  int flex_2;
  uint16_t crc;             // Cyclic redundancy check (CRC-16)
} BLEPacket;

/*---------------- Global variables ----------------*/

const unsigned int PACKET_SIZE = 20;
const unsigned int PKT_THRESHOLD = 5;
const int default_data[] = {0, 0, 0, 0, 0, 0, 0, 0};
const int shot_data[] = {1, 0, 0, 0, 0, 0, 0, 0};

static unsigned int health = 100;
static unsigned int shotCount = 0;
static unsigned int seqNo = 1;
static unsigned int counter = 0;

uint8_t serial_buffer[PACKET_SIZE];
BLEPacket* curr_packet;

/*---------------- CRC calculation ----------------*/

uint16_t crcCalc(uint8_t* data)
{
   uint16_t curr_crc = 0x0000;
   uint8_t sum1 = (uint8_t) curr_crc;
   uint8_t sum2 = (uint8_t) (curr_crc >> 8);

   for (int i = 0; i < PACKET_SIZE; i++)
   {
      sum1 = (sum1 + data[i]) % 255;
      sum2 = (sum2 + sum1) % 255;
   }
   return (sum2 << 8) | sum1;
}

/*---------------- Checks ----------------*/

bool crcCheck()
{
  uint16_t crc = curr_packet->crc;
  curr_packet->crc = 0;
  return (crc == crcCalc((uint8_t*)curr_packet));
}

bool packetCheck(uint8_t node_id, PacketType packet_type)
{
  uint8_t header = curr_packet->header;
  uint8_t curr_node_id = (header & 0xf0) >> 4;
  PacketType curr_packet_type = PacketType((header & 0b1100) >> 2);
  return curr_node_id == node_id && curr_packet_type == packet_type;
}

bool seqNoCheck()
{
  uint8_t header = curr_packet->header;
  uint8_t curr_seq_no = header & 0b1;
  return curr_seq_no != seqNo;
}

/*---------------- Packet management ----------------*/


BLEPacket generatePacket(PacketType packet_type, int* data)
{
  BLEPacket p;
  p.header = (2 << 4) | (packet_type << 2) | seqNo;
  p.padding = 0;
  p.euler_x = data[0];
  p.euler_y = data[1];
  p.euler_z = data[2];
  p.acc_x = data[3];
  p.acc_y = data[4];
  p.acc_z = data[5];
  p.flex_1 = data[6];
  p.flex_2 = data[7];
  p.crc = 0;
  uint16_t calculatedCRC = crcCalc((uint8_t*)&p);
  p.crc = calculatedCRC;
  return p;
}

void sendPacket(PacketType packet_type, int* data)
{
  BLEPacket p = generatePacket(packet_type, data);
  Serial.write((byte*)&p, PACKET_SIZE);
}

void sendDefaultPacket(PacketType packet_type)
{
  sendPacket(packet_type, default_data);
}

void sendDataPacket()
{
  int data[] = {counter, 0, 0, 0, 0, 0, 0, 0};
  sendPacket(DATA, data);
}

/*---------------- Game state handler ----------------*/

void updateGameState()
{
  health = curr_packet->euler_y;
}

/*---------------- Communication protocol ----------------*/

void waitForData()
{
  unsigned int buf_pos = 0;
  while (buf_pos < PACKET_SIZE)
  {
    if (Serial.available())
    {
      uint8_t in_data = Serial.read();
      serial_buffer[buf_pos] = in_data;
      buf_pos++;
    }
  }
  curr_packet = (BLEPacket*)serial_buffer;
}

void threeWayHandshake()
{
  bool is_connected = false;
  while (!is_connected)
  {
    // wait for hello from laptop
    waitForData();
  
    if (!crcCheck() || !packetCheck(0, HELLO))
    {
      sendDefaultPacket(NACK);
      continue;
    } 
    sendDefaultPacket(HELLO);

    // reset seq no
    seqNo = 1;

    shotCount = 0;
    
    // wait for ack from laptop
    waitForData();
    
    if (crcCheck() && packetCheck(0, ACK))
    {
      updateGameState();
      is_connected = true;
    }
  }
}

/*---------------- Setup ----------------*/

void setup() {
  Serial.begin(115200);
  threeWayHandshake();
}

/*---------------- Main ----------------*/

unsigned int count = 0;

void loop() {
  delay(100);
  count++;
  count %= 10;
  if (count == 0) shotCount++;
  if (shotCount > 0)
  {
    // increment sequence number for next packet
    seqNo++;
    seqNo %= 2;

    // initialize loop variables
    unsigned int pkt_count = 0;
    bool is_ack = false;

    while (!is_ack)
    {
      // only send data packet if the number of received ACKs has exceeded threshold
      if (pkt_count == 0) sendDataPacket();

      // receive and buffer serial data
      waitForData();

      // increment packet count
      pkt_count++;
      pkt_count %= PKT_THRESHOLD;
      
      // do checks on received data
      if (!crcCheck()) continue;
      if (packetCheck(0, ACK) && seqNoCheck())
      {
        shotCount--;
        counter++;
        updateGameState();
        is_ack = true;
      }
      else if (packetCheck(0, HELLO)) // reinitiate 3-way handshake
      {
        sendDefaultPacket(HELLO);

        // reset seq no
        seqNo = 1;

        shotCount = 0;
        
        // wait for ack from laptop
        waitForData();
        
        if (crcCheck() && packetCheck(0, ACK))
        {
          updateGameState();
          is_ack = true;
        }
      }
    }
  }
  else
  {
    waitForData();
    if (!crcCheck()) return;
    if (packetCheck(0, HELLO)) // reinitiate 3-way handshake
    {
      sendDefaultPacket(HELLO);

      // reset seq no
      seqNo = 1;

      shotCount = 0;
      
      // wait for ack from laptop
      waitForData();
      
      if (crcCheck() && packetCheck(0, ACK))
      {
        updateGameState();
      }
    }
    else if (packetCheck(0, ACK) && seqNoCheck()) // game state broadcast
    {
      updateGameState();
    }
  }
}
