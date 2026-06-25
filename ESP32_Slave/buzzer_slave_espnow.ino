// ================================================================
// buzzer_slave_espnow.ino
// ESP32 Slave — Primeste alarme via ESP-NOW si actioneaza buzzerul
//
// Conexiuni:
//   GPIO 27 → Buzzer pasiv (+ rezistor 100Ω in serie)
//   GND     → Buzzer GND
//
// Flux de utilizare:
//   1. Incarca acest sketch pe slave
//   2. Deschide Serial Monitor
//   3. Noteaza adresa MAC afisata la pornire
//   4. Pune acea adresa in SLAVE_MAC[] din fisierul master
//   5. Incarca masterul si reporneste ambele placi
// ================================================================

#include <esp_now.h>
#include <WiFi.h>

#define BUZZER_PIN   27
#define BUZZER_FREQ  2000   // Hz — frecventa ton (ajusteaza dupa buzzer)

// ── Structura pachet (identica cu masterul) ───────────────────
typedef struct {
    char cmd[8];   // "FOC", "FAULT", "NORMAL"
} AlarmPacket;

// ── Buffer comanda primita (setat din callback, executat in loop) ─
volatile bool newCmd   = false;
char pendingCmd[8]     = "";

// ================================================================
// Pattern buzzer — FOC
// 3 reprize × 3 bipuri scurte rapide (80ms on / 100ms off)
// Pauza 400ms intre reprize
// ================================================================
void beep(uint32_t ms) {
    tone(BUZZER_PIN, BUZZER_FREQ, ms);
    delay(ms);
    noTone(BUZZER_PIN);
}

void alarmFoc() {
    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < 3; i++) {
            beep(80);
            delay(100);
        }
        delay(400);
    }
}

// ================================================================
// Pattern buzzer — FAULT
// 2 reprize × 2 bipuri lungi (600ms on / 300ms off)
// Pauza 800ms intre reprize
// ================================================================
void alarmFault() {
    for (int r = 0; r < 2; r++) {
        beep(600);
        delay(300);
        beep(600);
        delay(800);
    }
}

// ================================================================
// Callback ESP-NOW — apelat la primirea unui pachet
// NU rula tone()/delay() direct din callback (context WiFi task)
// → Seteaza flag + copiaza comanda, executa in loop()
// ================================================================
void onDataReceived(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
    if (len != sizeof(AlarmPacket)) return;

    AlarmPacket pkt;
    memcpy(&pkt, data, sizeof(pkt));
    pkt.cmd[sizeof(pkt.cmd) - 1] = '\0';  // siguranta null-terminator

    strncpy(pendingCmd, pkt.cmd, sizeof(pendingCmd) - 1);
    pendingCmd[sizeof(pendingCmd) - 1] = '\0';
    newCmd = true;
}

// ================================================================
// SETUP
// ================================================================
void printMAC() {
    Serial.println("=========================================");
    Serial.println("[Slave] MAC Address: " + WiFi.macAddress());
    Serial.println("Copiaza MAC-ul de mai sus in SLAVE_MAC[] din master!");
    Serial.println("=========================================");
}

void setup() {
    Serial.begin(460800);

    // Clipeste LED-ul built-in 3s ca sa ai timp sa deschizi Serial Monitor
    pinMode(2, OUTPUT);
    for (int i = 0; i < 6; i++) {
        digitalWrite(2, !digitalRead(2));
        delay(500);
    }

    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    printMAC();

    if (esp_now_init() != ESP_OK) {
        Serial.println("[ESP-NOW] Init ESUAT — verifica placa!");
        return;
    }

    esp_now_register_recv_cb(onDataReceived);
    Serial.println("[ESP-NOW] Slave gata, astept comenzi...");

    beep(100); delay(100); beep(100);
}

// ================================================================
// LOOP — executa comanda primita (event-driven)
// ================================================================
void loop() {
    // Repeta MAC-ul la fiecare 5s pana primeste prima comanda de la master
    static bool paired = false;
    if (!paired) {
        static uint32_t lastPrint = 0;
        if (millis() - lastPrint > 5000) {
            printMAC();
            lastPrint = millis();
        }
    }

    if (newCmd) {
        paired = true;
        newCmd = false;

        Serial.print("[CMD] Primit: ");
        Serial.println(pendingCmd);

        if (strcmp(pendingCmd, "FOC") == 0) {
            Serial.println("[!] Alarma FOC — 3x3 bipuri scurte");
            alarmFoc();
        }
        else if (strcmp(pendingCmd, "FAULT") == 0) {
            Serial.println("[!] Alarma FAULT — bipuri lungi");
            alarmFault();
        }
        else if (strcmp(pendingCmd, "NORMAL") == 0) {
            // Stare normala — fara sunet
            Serial.println("[OK] Stare normala");
        }
        else {
            Serial.print("[?] Comanda necunoscuta: ");
            Serial.println(pendingCmd);
        }
    }

    delay(10);
}
