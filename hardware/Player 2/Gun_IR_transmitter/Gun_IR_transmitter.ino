// ================================================================
// ===               INTERNAL COMMS CODE              ===
// ================================================================
#pragma pack(1)
// ================================================================
// ===               LIBRARIES + VARIABLES              ===
// ================================================================
#include <Arduino.h>
#include <IRremote.hpp>

#define PUSHBUTTON_PIN  2
#define IR_SEND_PIN         3

byte segPins[] = {A3, A4, 4, 5, A5, A2, A1}; //7-seg LED segment display

// Variables will change:
int buttonStateNew;     // the current reading from the input pin
int actualButtonState;             // the current reading from the input pin
int lastButtonState = LOW;   // the previous reading from the input pin
unsigned long lastDebounceTime = 0;  // the last time the output pin was toggled
unsigned long debounceDelay = 50;    // the debounce time; increase if the output flickers

uint16_t sAddress = 0x0102;
// ================================================================
// ===               INTERNAL COMMS CODE              ===
// ================================================================

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
  uint16_t crc;             // Cyclic redundancy check (CRC-16)
} BLEPacket;

/*---------------- Global variables ----------------*/

const unsigned int PACKET_SIZE = 16;
const unsigned int PKT_THRESHOLD = 10;
const int default_data[] = {0, 0, 0, 0, 0, 0};
const int shoot_data[] = {1, 0, 0, 0, 0, 0};

static unsigned int bullets = 6;
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
  p.header = (4 << 4) | (packet_type << 2) | seqNo;
  p.padding = 0;
  p.euler_x = data[0];
  p.euler_y = data[1];
  p.euler_z = data[2];
  p.acc_x = data[3];
  p.acc_y = data[4];
  p.acc_z = data[5];
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
  int data[] = {counter, 0, 0, 0, 0, 0};
  sendPacket(DATA, data);
}

/*---------------- Game state handler ----------------*/

void updateGameState()
{
  bullets = curr_packet->euler_x;
}

/*---------------- Communication protocol ----------------*/

void waitForData()
{
  unsigned int buf_pos = 0;
  while (buf_pos < PACKET_SIZE)
  {
    if (Serial.available())
    {
      uint8_t in_data = Serial.read(); //keeps reading incoming data from laptop till pkt is completely constructed
      serial_buffer[buf_pos] = in_data;
      buf_pos++;
    }
  }
  curr_packet = (BLEPacket*)serial_buffer; //store current incoming data pkt
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
    
    // wait for ack from laptop
    waitForData();
    
    if (crcCheck() && packetCheck(0, ACK))
    {
      updateGameState();
      is_connected = true;
    }
  }
}

// ================================================================
// ===               SETUP CODE              ===
// ================================================================
void setup() {
  
  Serial.begin(115200);
  
  //enable button:
  pinMode(PUSHBUTTON_PIN, INPUT);
                  
  //enable IR LED:
  pinMode(IR_SEND_PIN, OUTPUT);  
  IrSender.begin(IR_SEND_PIN, ENABLE_LED_FEEDBACK, USE_DEFAULT_FEEDBACK_LED_PIN); //enable IR LED

  //enable LED 7 seg display:
  pinMode(13,OUTPUT);
  for(int i =0; i< 7; i++) {
    pinMode(segPins[i], OUTPUT);
  }
  
  //initialise led seg display to display 6 bullets:
  sevsegSetNumber(6);

  // int comms set up
  threeWayHandshake();
}


// ================================================================
// ===               FUNCTIONS            ===
// ================================================================
void display_0(){
      digitalWrite(segPins[0], LOW);
      digitalWrite(segPins[1], LOW);
      digitalWrite(segPins[2], LOW);
      digitalWrite(segPins[3], LOW);  
      digitalWrite(segPins[4], LOW);
      digitalWrite(segPins[5], LOW);      
      digitalWrite(segPins[6], HIGH);   
}
void display_1(){
      digitalWrite(segPins[1], LOW);
      digitalWrite(segPins[2], LOW);
      digitalWrite(segPins[0], HIGH);
      digitalWrite(segPins[3], HIGH);  
      digitalWrite(segPins[4], HIGH);
      digitalWrite(segPins[5], HIGH);      
      digitalWrite(segPins[6], HIGH);   
}
void display_2(){
      digitalWrite(segPins[2], HIGH);
      digitalWrite(segPins[5], HIGH);
      digitalWrite(segPins[0], LOW);
      digitalWrite(segPins[1], LOW);  
      digitalWrite(segPins[3], LOW);
      digitalWrite(segPins[4], LOW);      
      digitalWrite(segPins[6], LOW);  
}
void display_3(){
      digitalWrite(segPins[5], HIGH);
      digitalWrite(segPins[4], HIGH);
      digitalWrite(segPins[1], LOW);
      digitalWrite(segPins[0], LOW);
      digitalWrite(segPins[2], LOW);
      digitalWrite(segPins[3], LOW);  
      digitalWrite(segPins[6], LOW);
}
void display_4(){
      digitalWrite(segPins[0], HIGH);
      digitalWrite(segPins[4], HIGH);
      digitalWrite(segPins[3], HIGH);
      digitalWrite(segPins[1], LOW);  
      digitalWrite(segPins[2], LOW);
      digitalWrite(segPins[5], LOW);      
      digitalWrite(segPins[6], LOW);    
}
void display_5(){
      digitalWrite(segPins[1], HIGH);
      digitalWrite(segPins[4], HIGH);
      digitalWrite(segPins[0], LOW);
      digitalWrite(segPins[2], LOW);  
      digitalWrite(segPins[3], LOW);
      digitalWrite(segPins[5], LOW);      
      digitalWrite(segPins[6], LOW);        
}
void display_6(){
      digitalWrite(segPins[1], HIGH);
      digitalWrite(segPins[0], LOW);
      digitalWrite(segPins[2], LOW);
      digitalWrite(segPins[3], LOW);  
      digitalWrite(segPins[4], LOW);
      digitalWrite(segPins[5], LOW);      
      digitalWrite(segPins[6], LOW);   
}
void sevsegSetNumber(int num){
  if (num == 0) {
    display_0();
  }
  if (num == 1) {
    display_1();
  }
  if (num == 2) {
    display_2();
  }
  if (num == 3){
    display_3();
  }
  if (num == 4){
    display_4();
  }
  if (num == 5){
    display_5();
  }
  if (num == 6){
    display_6();
  }
}

void toggle_Ammo_display(){
  /*
    if(ammo > 0) {
      ammo -= 1; //minus one for ammo and display this number on led segment display + send this info to visualiser
    }
    //ammo = 0; if user reload, reset ammo to 6
    else {
      ammo = 6;
    } 
    */
    sevsegSetNumber(bullets);
}

void send_Shoot_Packet(){
    
    // increment sequence number for next packet - ie. seqNo 0 and 1 only
    seqNo++;
    seqNo %= 2;

    // initialize loop variables
    unsigned int pkt_count = 0;
    bool is_ack = false;

    while (!is_ack)
    {
      // send next data pkt
      // or only resend current data packet if the number of failed CRC checks has exceeded threshold = 5 OR no ACK  OR HELLO pkt received from laptop 
      if (pkt_count == 0) sendDataPacket();

      // receive and buffer serial data
      waitForData();

      // increment packet count
      pkt_count++;
      pkt_count %= PKT_THRESHOLD;
      
      // do checks on received data
      if (!crcCheck()) continue;
      //Serial.print(seqNoCheck());
      if (packetCheck(0, ACK) && seqNoCheck())
      {
        counter++; //tracks no of shots 
        updateGameState(); //receiving ACK with bullet count from SW -> need to sync with beetle
        toggle_Ammo_display();
        is_ack = true;
      }
      else if (packetCheck(0, HELLO)) // reinitiate 3-way handshake - after disconnection
      {
        sendDefaultPacket(HELLO);

        // reset seq no
        seqNo = 1;
        
        // wait for ack from laptop
        waitForData();
        
        if (crcCheck() && packetCheck(0, ACK))
        {
          updateGameState();
          toggle_Ammo_display();
        }
      }
    }
    
}

void wait_Relay_Node(){
  if(Serial.available()){
        waitForData();
      if (!crcCheck()) return;
      if (packetCheck(0, HELLO)) // reinitiate 3-way handshake
      {
        sendDefaultPacket(HELLO);
    
        // reset seq no
        seqNo = 1;
        
        // wait for ack from laptop
        waitForData();
        
        if (crcCheck() && packetCheck(0, ACK))
        {
          updateGameState(); //re-read bullets from laptop(SW) -> ARDUINO SYNC WITH GAMESTATE FROM SW
          toggle_Ammo_display();
        }
      }
      else if (packetCheck(0, ACK) && seqNoCheck()) // game state broadcast (when laptop SENDS ACK)
      {
        updateGameState();
        toggle_Ammo_display();
      }
  }
  
  
}

// ================================================================
// ===               MAIN LOOP              ===
// ================================================================
void loop() {
  
  //delay(100);
  
  // ===               MAIN CODE              ===
   // read the state of the switch into a local variable:
  buttonStateNew = digitalRead(PUSHBUTTON_PIN);

  // If the switch changed, due to noise or pressing/release:
  if (buttonStateNew != lastButtonState) {
    // reset the debouncing timer
    lastDebounceTime = millis();
  }
  // whatever the reading is at, it's been there for longer than the debounce
  // delay, so take it as the actual current state:
  if ((millis() - lastDebounceTime) > debounceDelay) {

    // if the button state has changed:
    if (buttonStateNew != actualButtonState) {
      actualButtonState = buttonStateNew;

      // only send IR LED if the new button state is HIGH
      if (actualButtonState == HIGH) { 
        IrSender.sendNEC(sAddress, 0x01, 0); //Command sent: 0x01
        send_Shoot_Packet(); // data pkt sent to SW Visualiser
      }
      else {
        wait_Relay_Node();
      }
    }
    else {
      wait_Relay_Node();
    }
    //either: time limit has not passed 50ms and gun shot
    //or: no shooting action
  }
  else {
    wait_Relay_Node(); 
  }
  
  // save the reading. Next time through the loop, it'll be the lastButtonState:
  lastButtonState = buttonStateNew;

  // ===               END              ===
}
