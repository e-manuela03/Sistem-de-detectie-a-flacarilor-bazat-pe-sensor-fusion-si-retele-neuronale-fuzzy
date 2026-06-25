#include <Wire.h>
#include <Adafruit_MLX90640.h>

Adafruit_MLX90640 mlx;
float frame[32 * 24];

//  USER SETTINGS 
const int IR_PIN = 35;
//const int UV_PIN = 34;
const int NUM_SAMPLES_IR = 50;     // averaging for IR
//const int NUM_SAMPLES_UV = 20;     // averaging for UV if you want later
const bool SEND_FULL_FRAME = true; // true = send all 768 values
const unsigned long LOOP_DELAY_MS = 50;

int readAveragedADC(int pin, int nSamples) {
  long sum = 0;
  for (int i = 0; i < nSamples; i++) {
    sum += analogRead(pin);
    delay(2);
  }
  return sum / nSamples;
}

void setup() {
  Serial.begin(460800); //i set higher baud rate to fix lagging
  delay(1000);

  // ADC config
  analogReadResolution(12); // 0-4095
  analogSetPinAttenuation(IR_PIN, ADC_11db);
  //analogSetPinAttenuation(UV_PIN, ADC_2_5db);

  // MLX90640 init
  Wire.begin(21, 22);
  Wire.setClock(400000);

  if (!mlx.begin()) {
    Serial.println("ERROR,MLX90640_NOT_FOUND");
    while (1);
  }

  mlx.setMode(MLX90640_CHESS);
  mlx.setResolution(MLX90640_ADC_18BIT);
  mlx.setRefreshRate(MLX90640_8_HZ);

  if (SEND_FULL_FRAME) {
    Serial.println("READY,format=ts,ir_raw,ir_avg,tmin,tmax,tmean,frame768");
  } else {
    Serial.println("READY,format=ts,ir_raw,ir_avg,tmin,tmax,tmean");
  }
}

void loop() {
  // Read analog sensors
  unsigned long ts = millis();

  int irRaw = 4095-analogRead(IR_PIN);
  int irAvg_raw = readAveragedADC(IR_PIN, NUM_SAMPLES_IR);
  int irAvg=4095 - irAvg_raw; 
  //int uvRaw = analogRead(UV_PIN);

  // Read thermal frame
  if (mlx.getFrame(frame) != 0) {
    Serial.println("ERROR,FRAME_ERROR");
    delay(200);
    return;
  }

  float minT = frame[0];
  float maxT = frame[0];
  float sumT = 0.0;

  for (int i = 0; i < 768; i++) {
    if (frame[i] < minT) minT = frame[i];
    if (frame[i] > maxT) maxT = frame[i];
    sumT += frame[i];
  }

  float meanT = sumT / 768.0;

  // Unified serial output
  Serial.print(ts);
  Serial.print(",");
  Serial.print(irRaw);
  Serial.print(",");
  Serial.print(irAvg);
  Serial.print(",");
 // Serial.print(uvRaw);
 // Serial.print(",");
  Serial.print(minT, 2);
  Serial.print(",");
  Serial.print(maxT, 2);
  Serial.print(",");
  Serial.print(meanT, 2);

  if (SEND_FULL_FRAME) {
    for (int i = 0; i < 768; i++) {
      Serial.print(",");
      Serial.print(frame[i], 2);
    }
  }

  Serial.println();

  delay(LOOP_DELAY_MS);
}