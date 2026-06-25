// ================================================================
// anfis_inference_display.ino  — cu ESP-NOW (Master)
// ANFIS 3 intrari + feed termic pe ST7735S + LED rosu/verde/galben
// + ESP-NOW → trimite "FOC" / "FAULT" / "NORMAL" la slave buzzer
//
// !!! OBLIGATORIU inainte de upload: !!!
//   1. Incarca mai intai buzzer_slave_espnow.ino pe slave
//   2. Deschide Serial Monitor pe slave, citeste MAC-ul afisat
//   3. Completeaza SLAVE_MAC[] de mai jos cu acel MAC
// ================================================================

#include <Wire.h>
#include <Adafruit_MLX90640.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>
#include <SPI.h>
#include <math.h>
#include <esp_now.h>
#include <WiFi.h>
#include "anfis_params.h"

// FIX 1: enum declarat IMEDIAT dupa includes.
// Arduino IDE genereaza prototipuri automat si le insereaza la inceputul
// fisierului — inaintea oricarui cod. Daca enum-ul e mai jos, prototipul
// "FaultType updateIRFault(...)" apare inainte ca tipul sa fie cunoscut.
enum FaultType { FAULT_NONE, FAULT_STUCK, FAULT_ROC, FAULT_MLX };

// ── Pini ST7735S ─────────────────────────────────────────────
#define TFT_CS    5
#define TFT_DC    4
#define TFT_RST   2

Adafruit_ST7735 tft = Adafruit_ST7735(TFT_CS, TFT_DC, TFT_RST);

// ── Pini senzori si LED ──────────────────────────────────────
#define IR_PIN          35
#define LED_RED         17
#define LED_GREEN       26
#define LED_YELLOW      27
#define FIRE_THRESHOLD  0.55f
#define NUM_SAMPLES_IR  10
#define LOOP_DELAY_MS   50

// ── Limite fizice senzori pentru sanitizare frame ─────────────
#define TEMP_MIN_VALID  0.0f
#define TEMP_MAX_VALID  460.0f
#define TEMP_FALLBACK   25.0f

// ================================================================
// ESP-NOW — configuratie
// ================================================================

uint8_t SLAVE_MAC[] = { 0xC8, 0xF0, 0x9E, 0x4F, 0x46, 0x64 };

typedef struct {
    char cmd[8];   // "FOC", "FAULT", "NORMAL"
} AlarmPacket;

AlarmPacket alarmPkt;

// FIX 2: callback-ul onSent eliminat.
// In ESP32 Arduino core 3.x semnatura s-a schimbat din
//   void cb(const uint8_t*, esp_now_send_status_t)
// in
//   void cb(const wifi_tx_info_t*, esp_now_send_status_t)
// Callback-ul era doar debug; functionalitatea nu e afectata.

void espnowInit() {
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    if (esp_now_init() != ESP_OK) {
        Serial.println("[ESP-NOW] Init ESUAT");
        return;
    }

    esp_now_peer_info_t peer = {};
    memcpy(peer.peer_addr, SLAVE_MAC, 6);
    peer.channel = 0;
    peer.encrypt = false;

    if (esp_now_add_peer(&peer) != ESP_OK)
        Serial.println("[ESP-NOW] Add peer ESUAT");
    else
        Serial.println("[ESP-NOW] Peer slave adaugat OK");
}

void sendAlarmIfChanged(const char* cmd) {
    static char lastCmd[8] = "INIT";
    if (strcmp(cmd, lastCmd) != 0) {
        strncpy(alarmPkt.cmd, cmd, sizeof(alarmPkt.cmd) - 1);
        alarmPkt.cmd[sizeof(alarmPkt.cmd) - 1] = '\0';
        esp_now_send(SLAVE_MAC, (uint8_t*)&alarmPkt, sizeof(alarmPkt));
        strncpy(lastCmd, cmd, sizeof(lastCmd) - 1);
        Serial.print("[ESP-NOW] Trimis -> ");
        Serial.println(cmd);
    }
}

// ================================================================
// DETECTIE FAULT SENZOR IR — doua criterii
//
//  FAULT_ROC  : contor leaky pe delta intre citiri consecutive
//               Citire rea  (|delta| > ROC_THR) : cnt += ROC_INC
//               Citire buna (|delta| <= ROC_THR) : cnt -= ROC_DEC
//               FAULT daca comportamentul rau e SUSTINUT ~10s
//               → prinde senzor deconectat cu pin flotant haotic
//
//  FAULT_STUCK: IR_STUCK_CONSEC citiri consecutive de exact 0
//               sau exact 4095 (railuri ADC)
//               → prinde senzor anchorat ferm la GND sau VCC
//               → NU se declanseaza pe senzor normal noaptea
//                 (noaptea citeste valori stabile dar NU la exact
//                  0/4095 — zgomotul ADC le tine la 1-30 / 4065-4094)
// ================================================================

// ── ROC leaky counter ─────────────────────────────────────────
#define IR_ROC_THR          400.0f
#define IR_ROC_INC            1.0f
#define IR_ROC_DEC            0.5f
#define IR_ROC_FAULT_COUNT  200.0f

// ── STUCK la rail exact ───────────────────────────────────────
// 200 citiri × 50ms = 10 secunde continuu la 0 sau 4095
#define IR_STUCK_CONSEC_MAX 200

// (enum FaultType declarat la inceputul fisierului)

float ir_prev      = -1.0f;
float ir_roc_cnt   =  0.0f;
int   ir_stuck_cnt =  0;

FaultType fault_type = FAULT_NONE;

// ── Camera termica ───────────────────────────────────────────
Adafruit_MLX90640 mlx;
float frame[32 * 24];

// ── Layout display (160x128 landscape) ──────────────────────
#define PIXEL_W   4
#define PIXEL_H   5
#define FRAME_X   0
#define FRAME_Y   0
#define INFO_Y   122

// ── Paleta inferno (16 culori RGB565) ───────────────────────
const uint16_t INFERNO[] = {
    0x0000, 0x200A, 0x4013, 0x600B,
    0x8912, 0xA802, 0xC880, 0xE8E0,
    0xF9E0, 0xFAE0, 0xFBE0, 0xFCC0,
    0xFDE0, 0xFF00, 0xFF80, 0xFFFF
};
#define PALETTE_SIZE 16

float T_MIN = 20.0f;
float T_MAX = 40.0f;

// ================================================================
// FILTRU KALMAN 1D
// ================================================================
struct KalmanFilter {
    float process_variance;
    float measurement_variance;
    float estimate;
    float estimate_error;
    bool  initialized;

    KalmanFilter(float pv, float mv)
        : process_variance(pv), measurement_variance(mv),
          estimate(0.0f), estimate_error(1.0f), initialized(false) {}

    float update(float z) {
        if (!initialized) { estimate = z; initialized = true; }
        float pred_error = estimate_error + process_variance;
        float K          = pred_error / (pred_error + measurement_variance);
        estimate         = estimate + K * (z - estimate);
        estimate_error   = (1.0f - K) * pred_error;
        return estimate;
    }
};
KalmanFilter kf_ir_raw(0.8f,  20.0f);
KalmanFilter kf_tmax  (1e-5f,  1.0f);
KalmanFilter kf_tmean (1e-5f,  1.0f);

// ================================================================
// ANFIS — inferenta
// ================================================================
float normalize(float val, float mn, float mx) {
    val = fmaxf(mn, fminf(mx, val));
    return (val - mn) / (mx - mn + 1e-9f);
}

float gaussMF(float x, float sigma, float c) {
    return expf(-powf(x - c, 2.0f) / (2.0f * sigma * sigma + 1e-9f));
}

float anfisInfer(float ir_raw_k, float tmax_k, float tmean_k) {
    float x[ANFIS_N_INPUTS] = {
        normalize(ir_raw_k,  ANFIS_MINS[0], ANFIS_MAXS[0]),
        normalize(tmax_k,    ANFIS_MINS[1], ANFIS_MAXS[1]),
        normalize(tmean_k,   ANFIS_MINS[2], ANFIS_MAXS[2])
    };
    float mu[ANFIS_N_INPUTS][ANFIS_N_MF];
    for (int i = 0; i < ANFIS_N_INPUTS; i++)
        for (int k = 0; k < ANFIS_N_MF; k++)
            mu[i][k] = gaussMF(x[i], ANFIS_SIGMA[i][k], ANFIS_C[i][k]);

    float w[ANFIS_N_RULES];
    for (int r = 0; r < ANFIS_N_RULES; r++) {
        w[r] = 1.0f;
        for (int i = 0; i < ANFIS_N_INPUTS; i++)
            w[r] *= mu[i][ANFIS_RULES[r][i]];
    }
    float wSum = 0.0f;
    for (int r = 0; r < ANFIS_N_RULES; r++) wSum += w[r];
    if (wSum < ANFIS_WSUM_GUARD) return 0.0f;

    float output = 0.0f;
    for (int r = 0; r < ANFIS_N_RULES; r++)
        output += (w[r] / (wSum + 1e-9f)) * ANFIS_P[r];

    if (isnan(output) || isinf(output)) return 0.0f;
    return fmaxf(0.0f, fminf(1.0f, output));
}

// ================================================================
// DETECTIE FAULT — functii
// ================================================================
FaultType updateIRFault(float ir_val, bool fire_active) {
    if (fault_type != FAULT_NONE) return fault_type;  // latch

    // ── Garda foc activ ───────────────────────────────────────
    // Cand ANFIS detecteaza foc, IR-ul poate satura sau oscila
    // din cauza flamei — nu din cauza unui defect. Inghetam
    // contorii ca sa nu transformam o alarma reala in FAULT.
    // ir_prev se actualizeaza totusi ca sa nu explodeze delta
    // la prima citire dupa ce focul se stinge.
    if (fire_active) {
        ir_prev = ir_val;
        return FAULT_NONE;
    }

    // ── Criteriul ROC — contor leaky ─────────────────────────
    // Citiri haotice (delta mare) acumuleaza contorul.
    // Citiri calme il erodeaza. FAULT doar daca haosul e sustinut.
    if (ir_prev >= 0.0f) {
        float delta = fabsf(ir_val - ir_prev);
        if (delta > IR_ROC_THR)
            ir_roc_cnt = fminf(ir_roc_cnt + IR_ROC_INC, IR_ROC_FAULT_COUNT);
        else
            ir_roc_cnt = fmaxf(ir_roc_cnt - IR_ROC_DEC, 0.0f);
    }
    ir_prev = ir_val;

    if (ir_roc_cnt > 0.0f) {
        static uint32_t last_log = 0;
        if (millis() - last_log > 500) {
            Serial.print("  [ROC cnt=");
            Serial.print(ir_roc_cnt, 1);
            Serial.print("/");
            Serial.print(IR_ROC_FAULT_COUNT, 0);
            Serial.println("]");
            last_log = millis();
        }
    }

    if (ir_roc_cnt >= IR_ROC_FAULT_COUNT) return FAULT_ROC;

    // ── Criteriul STUCK — rail exact ─────────────────────────
    // Senzorul conectat normal produce zgomot ADC: nu ajunge
    // niciodata la 0 sau 4095 exact, nici noaptea, nici ziua.
    // Un pin la GND / VCC sau un fir rupt anchorat citeste
    // valoarea de rail perfect, fara abatere, sute de iteratii.
    int iv = (int)ir_val;
    if (iv == 0 || iv == 4095)
        ir_stuck_cnt++;
    else
        ir_stuck_cnt = 0;

    if (ir_stuck_cnt >= IR_STUCK_CONSEC_MAX) return FAULT_STUCK;

    return FAULT_NONE;
}

void drawFaultScreen(float ir_val, FaultType ft) {
    tft.fillScreen(0xFFE0);
    tft.setTextColor(ST77XX_BLACK);
    tft.setTextSize(3);
    tft.setCursor(18, 10);
    tft.print("FAULT");

    tft.drawFastHLine(0, 42, tft.width(), ST77XX_BLACK);
    tft.setTextSize(1);
    tft.setCursor(6, 50);
    if (ft == FAULT_ROC) {
        tft.print("IR: salturi sustinute!");
        tft.setCursor(6, 62);
        tft.print("(senzor deconectat?)");
    } else if (ft == FAULT_STUCK) {
        tft.print("IR blocat la rail!");
        tft.setCursor(6, 62);
        tft.print("(scurtcircuit GND/VCC)");
    } else {
        tft.print("Camera termica absenta!");
        tft.setCursor(6, 62);
        tft.print("(MLX90640 nedelectat)");
    }

    tft.setCursor(6, 78);
    tft.print("Valoare: ");
    tft.print((int)ir_val);
    tft.print(" LSB");

    tft.drawFastHLine(0, 90, tft.width(), ST77XX_BLACK);
    tft.setCursor(6, 98);
    tft.print("Verificati cablajul");
    tft.setCursor(6, 110);
    tft.print("si reporniti!");
}

// ================================================================
// DISPLAY — functii helper
// ================================================================
uint16_t tempToColor(float t) {
    float norm = (t - T_MIN) / (T_MAX - T_MIN + 1e-9f);
    norm = fmaxf(0.0f, fminf(1.0f, norm));
    return INFERNO[(int)(norm * (PALETTE_SIZE - 1))];
}

void drawThermalFrame(float* f, float minT, float maxT) {
    T_MIN = minT; T_MAX = maxT;
    for (int row = 0; row < 24; row++)
        for (int col = 0; col < 32; col++) {
            uint16_t c = tempToColor(f[row * 32 + col]);
            tft.fillRect(FRAME_X + col * PIXEL_W, FRAME_Y + row * PIXEL_H,
                         PIXEL_W, PIXEL_H, c);
        }
}

void drawTempScale(float minT, float maxT) {
    int barX = FRAME_X + 32 * PIXEL_W + 2;
    int barH = 24 * PIXEL_H;
    for (int i = 0; i < barH; i++) {
        float norm = 1.0f - (float)i / barH;
        tft.drawFastHLine(barX, FRAME_Y + i, 8,
                          INFERNO[(int)(norm * (PALETTE_SIZE - 1))]);
    }
    tft.setTextSize(1);
    tft.setTextColor(ST77XX_WHITE);
    tft.setCursor(barX, FRAME_Y);             tft.print((int)maxT);
    tft.setCursor(barX, FRAME_Y + barH - 8);  tft.print((int)minT);
}

void drawInfo(float ir_raw, float tmax_k, float tmean_k,
              float score, bool fire) {
    uint16_t bgColor = fire ? 0xF800 : ST77XX_BLACK;
    tft.fillRect(0, INFO_Y, tft.width(), tft.height() - INFO_Y, bgColor);

    tft.setTextSize(1);
    tft.setTextColor(ST77XX_WHITE);

    tft.setCursor(0, INFO_Y);
    tft.print("IR:"); tft.print((int)ir_raw);
    tft.setCursor(60, INFO_Y);
    tft.print("Tx:"); tft.print((int)tmax_k); tft.print("C");

    tft.setCursor(0, INFO_Y + 10);
    tft.print("Tm:"); tft.print((int)tmean_k); tft.print("C");
    tft.setCursor(60, INFO_Y + 10);
    tft.print("Sc:"); tft.print(score, 2);

    tft.setCursor(115, INFO_Y + 3);
    if (fire) { tft.setTextColor(ST77XX_WHITE); tft.print("FOC!"); }
    else       { tft.setTextColor(0x07E0);       tft.print("OK");  }
}

// ================================================================
// HELPER: citire IR cu mediere ADC
// ================================================================
int readAveragedADC(int pin, int nSamples) {
    long sum = 0;
    for (int i = 0; i < nSamples; i++) {
        sum += analogRead(pin);
        delayMicroseconds(200);
    }
    return sum / nSamples;
}

// ================================================================
// SETUP
// ================================================================
void setup() {
    Serial.begin(460800);
    delay(1000);

    espnowInit();
    Serial.print("[Master] MAC: ");
    Serial.println(WiFi.macAddress());

    pinMode(LED_RED,    OUTPUT);
    pinMode(LED_GREEN,  OUTPUT);
    pinMode(LED_YELLOW, OUTPUT);

    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_RED,    HIGH); digitalWrite(LED_GREEN, LOW);  digitalWrite(LED_YELLOW, LOW);  delay(300);
        digitalWrite(LED_RED,    LOW);  digitalWrite(LED_GREEN, HIGH); digitalWrite(LED_YELLOW, LOW);  delay(300);
        digitalWrite(LED_RED,    LOW);  digitalWrite(LED_GREEN, LOW);  digitalWrite(LED_YELLOW, HIGH); delay(300);
    }
    digitalWrite(LED_RED, LOW); digitalWrite(LED_GREEN, HIGH); digitalWrite(LED_YELLOW, LOW);

    tft.initR(INITR_BLACKTAB);
    tft.setRotation(1);
    tft.fillScreen(ST77XX_BLACK);
    tft.setTextColor(ST77XX_WHITE);
    tft.setTextSize(1);
    tft.setCursor(20, 55);
    tft.print("Initializare...");

    analogReadResolution(12);
    analogSetPinAttenuation(IR_PIN, ADC_11db);

    Wire.begin(21, 22);
    Wire.setClock(400000);

    if (!mlx.begin()) {
        Serial.println("ERROR,MLX90640_NOT_FOUND");
        fault_type = FAULT_MLX;   // loop() va afisa ecranul de fault si trimite ESP-NOW
    } else {
        mlx.setMode(MLX90640_CHESS);
        mlx.setResolution(MLX90640_ADC_18BIT);
        mlx.setRefreshRate(MLX90640_8_HZ);
    }

    tft.fillScreen(ST77XX_BLACK);
    Serial.println("READY,ANFIS_3INPUT_DISPLAY+ESPNOW");
    Serial.print("ROC: THR="); Serial.print(IR_ROC_THR, 0);
    Serial.print(" INC="); Serial.print(IR_ROC_INC, 1);
    Serial.print(" DEC="); Serial.print(IR_ROC_DEC, 1);
    Serial.print(" FAULT_AT="); Serial.println(IR_ROC_FAULT_COUNT, 0);
    Serial.print("STUCK: CONSEC="); Serial.println(IR_STUCK_CONSEC_MAX);
}

// ================================================================
// LOOP
// ================================================================
void loop() {

    float ir_raw = 4095 - readAveragedADC(IR_PIN, NUM_SAMPLES_IR);

    static bool last_fire = false;
    fault_type = updateIRFault(ir_raw, last_fire);

    if (fault_type != FAULT_NONE) {
        digitalWrite(LED_RED,    LOW);
        digitalWrite(LED_GREEN,  LOW);
        digitalWrite(LED_YELLOW, HIGH);
        drawFaultScreen(ir_raw, fault_type);
        sendAlarmIfChanged("FAULT");
        Serial.print("FAULT,");
        if      (fault_type == FAULT_ROC)   Serial.print("ROC");
        else if (fault_type == FAULT_STUCK) Serial.print("STUCK");
        else                                Serial.print("MLX");
        Serial.print(",val="); Serial.println((int)ir_raw);
        delay(1000);
        return;
    }

    if (mlx.getFrame(frame) != 0) {
        Serial.println("ERROR,FRAME_ERROR");
        delay(200);
        return;
    }

    float minT = 999.0f, maxT = -999.0f, sumT = 0.0f;
    int valid_count = 0;
    for (int i = 0; i < 768; i++) {
        float val = frame[i];
        if (isnan(val) || isinf(val) ||
            val < TEMP_MIN_VALID || val > TEMP_MAX_VALID) {
            frame[i] = TEMP_FALLBACK; val = TEMP_FALLBACK;
        }
        if (val < minT) minT = val;
        if (val > maxT) maxT = val;
        sumT += val;
        valid_count++;
    }
    float tmean = sumT / valid_count;

    float tmax_k   = kf_tmax.update(constrain(maxT,   20.0f, 450.0f));
    float tmean_k  = kf_tmean.update(constrain(tmean,  20.0f,  65.0f));
    float ir_raw_k = kf_ir_raw.update(constrain(ir_raw, 0.0f, 4095.0f));

    float score = anfisInfer(ir_raw_k, tmax_k, tmean_k);
    bool  fire  = (score >= FIRE_THRESHOLD);

    sendAlarmIfChanged(fire ? "FOC" : "NORMAL");

    drawThermalFrame(frame, minT, maxT);
    drawTempScale(minT, maxT);
    drawInfo(ir_raw, tmax_k, tmean_k, score, fire);

    digitalWrite(LED_RED,    fire ? HIGH : LOW);
    digitalWrite(LED_GREEN,  fire ? LOW  : HIGH);
    digitalWrite(LED_YELLOW, LOW);

    Serial.print("IR=");       Serial.print(ir_raw);
    Serial.print(" Tmax_K=");  Serial.print(tmax_k,  2);
    Serial.print(" Tmean_K="); Serial.print(tmean_k, 2);
    Serial.print(" Score=");   Serial.print(score,   3);
    Serial.print(" -> ");
    Serial.println(fire ? "FOC" : "NORMAL");

    last_fire = fire;

    delay(LOOP_DELAY_MS);
}
