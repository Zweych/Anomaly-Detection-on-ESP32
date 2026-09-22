#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>

#define I2C_SDA 21
#define I2C_SCL 22
Adafruit_BME280 bme;

void setup() {
    Serial.begin(115200);
    Wire.begin(I2C_SDA, I2C_SCL);

    if (!bme.begin(0x76, &Wire)) { 
        Serial.println("Error: Sensor BME280 tidak terdeteksi.");
        while (1);
    }
}

void loop() {
    float temp = bme.readTemperature();
    float press = bme.readPressure() / 100.0F; // Konversi ke hPa
    float hum = bme.readHumidity();

    // Format output: temperature,pressure,humidity
    Serial.print(temp);
    Serial.print(",");
    Serial.print(press);
    Serial.print(",");
    Serial.println(hum);

    delay(5000); // Interval perekaman
}