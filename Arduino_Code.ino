///////// IRON DOME //////////////


//Altitude motor name is Yaskawa and azimuth motor name is Fuji

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MCP23X17.h>
#include <Adafruit_MCP4725.h>
#include <freertos/FreeRTOS.h>
#include <freertos/Task.h>
#include <ESP32Servo.h>

Servo esc1, esc2, servo; // Declare the external Servo objects
unsigned long runStartTime = 0;

// Pin Definitions
#define SDA_I2C 12  // ESP32 I2C Data
#define SCL_I2C 13  // ESP32 I2C Clock
#define INTB 4 //Interupted pin of Expander
#define INTA 23 //Interupted pin of Expander 
#define Fuji_EN_A 36 // Azimuth motor encoder signal output (A)
#define Fuji_EN_B 39 // Azimuth motor encoder signal output (B)
#define Fuji_EN_Z 32 // Azimuth motor encoder signal output (Z)
#define Yas_EN_A 34 // Altitude motor encoder signal output (A)
#define Yas_EN_B 35 // Altitude motor encoder signal output (B)
#define Yas_EN_Z 33 // Altitude motor encoder signal output (Z)
#define Y_DAC_SCL 22 // DAC SCL for altitude motor 
#define Y_DAC_SDA 21 // DAC SDA for altitude motor
#define F_DAC_SCL 17 // DAC SCL for azimuth motor
#define F_DAC_SDA 16 // DAC SDA for azimuth motor
#define Y_PULSE 27 // Pulse command for position control of altitude motor
#define Y_DIR 14 // Direction command for position control of altitude motor
#define Y_CLR 15 // Clear command for position control of altitude motor
#define F_PULSE 25 // Pulse command for position control of azimuth motor
#define F_DIR 26 // Direction command for position control of azimuth motor
#define RELOAD_SERVO 5 // Mini servo SG90 for shooting
#define ESC1 18 // PWM operation for BLDC drive
#define ESC2 19 // PWM operation for BLDC drive

//For turn_to_target
#define AZIMUTH_TASK_PRIORITY 1
#define ALTITUDE_TASK_PRIORITY 1
#define SHOOT_TASK_PRIORITY 3 
#define STACK_SIZE 4096 // Adjust stack size as needed

// ESC Parameters
int BLAST_SPEED = 2000;
float compensate_Azi;

TaskHandle_t azimuthTaskHandle = NULL;
TaskHandle_t altitudeTaskHandle = NULL;
TaskHandle_t shootTaskHandle = NULL;
TaskHandle_t shotSchedulerTaskHandle = NULL; // <--- NEW: Handle for the task that schedules the shot

// *** MOVE THIS TYPEDEF STRUCT DEFINITION HERE ***
typedef struct {
    float targetAngle;
} MotorControlData;
typedef struct {
    int delayBeforeShootMs; 
} ShotSchedulerData;

// GLOBAL STATIC VARIABLES (Add these at the top of your .ino file, after the struct definition)
static MotorControlData azimuthGlobalData;
static MotorControlData altitudeGlobalData;
static ShotSchedulerData shotSchedulerGlobalData;

// MCP23017 and DAC Objects
Adafruit_MCP23X17 mcp;
Adafruit_MCP4725 dac_Tf; 

bool isInitialized = false;

// Variables for encoder positions
volatile int  fujiFinished = 1;
volatile int yaskawaFinished = 1;
volatile int fujiPosition = 0;
volatile int yasPosition = 0;
volatile int N_fujiPosition;
volatile int v;
volatile int k;
float F_deg_per_pulse = -72.0 / 4096.0;
float Y_deg_per_pulse = -45.0 / 4096.0;


volatile int lastFujiA = 0; //Initialize encoder for both servo motor
volatile int lastYasA = 0; 

// Function Prototypes
void handleCommand(String command);
void homeSetting();
void handleJog(String command);
void Gun(String command);
void setIO(String command);
void turn_to_target(String command); //New added function for quick turn to target
void generatePulses(int pulsePin, int pulseCount, int delay);

TwoWire WireF = TwoWire(1); // For Fuji DAC


// Interrupt Service Routine for Fuji Encoder (Azimuth Motor)
void IRAM_ATTR readFujiEncoder() {
    int currentA = digitalRead(Fuji_EN_A);
    int currentB = digitalRead(Fuji_EN_B);
    
    if (currentA != lastFujiA) { // Detect A signal change
        if (currentA == currentB) {
            fujiPosition++;  // Clockwise
        } else {
            fujiPosition--;  // Counter-clockwise
        }
    }
    lastFujiA = currentA;
}

// Interrupt Service Routine for Yaskawa Encoder (Altitude Motor)
void IRAM_ATTR readYasEncoder() {
    int currentA = digitalRead(Yas_EN_A);
    int currentB = digitalRead(Yas_EN_B);
    
    if (currentA != lastYasA) { // Detect A signal change
        if (currentA == currentB) {
            yasPosition++;  // Clockwise
        } else {
            yasPosition--;  // Counter-clockwise
        }
    }
    lastYasA = currentA;
}


void waitForEnter(const String& message) {
    Serial.println(message);
    while (!Serial.available()) {
      delay(10);
    }
    Serial.readStringUntil('\n');
  }

void generatePulses(int pulsePin, int pulseCount, int delay) {
    for (int i = 0; i < pulseCount; i++) {
        digitalWrite(pulsePin, HIGH);
        delayMicroseconds(delay); // slightly longer high time for more stability
        digitalWrite(pulsePin, LOW);
        delayMicroseconds(delay);
    }
}

void zero_orderhold_F(float voltage) { //Use for Speed command of Fuji servo motor (Azimuth motor)
    uint16_t dacValue = (uint16_t)(4095 * voltage / 5); 
    dac_Tf.setVoltage(dacValue, false);  // Update DAC Tf
}

bool isSwitchPressedStable(int pin, int requiredCount = 5, int sampleDelay = 5) {
    int confirmCount = 0;
    for (int i = 0; i < requiredCount + 3; i++) {
        if (mcp.digitalRead(pin) == 1) {
            confirmCount++;
            if (confirmCount >= requiredCount) return true;
        } else {
            confirmCount = 0;
        }
        delay(sampleDelay);
    }
    return false;
}

bool isSwitchReleasedStable(int pin, int requiredCount = 5, int sampleDelay = 5) {
    int confirmCount = 0;
    for (int i = 0; i < requiredCount + 3; i++) {
        if (mcp.digitalRead(pin) == 0) {
            confirmCount++;
            if (confirmCount >= requiredCount) return true;
        } else {
            confirmCount = 0;
        }
        delay(sampleDelay);
    }
    return false;
}

void setup() {
    Serial.begin(115200);
    delay(500); // Allow time for Serial to start
    Serial.println("Enter '55' to start MCP23017 setup...");

    //ESC PWM Setup
    esc1.attach(ESC1, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
    esc2.attach(ESC2, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
    servo.attach(RELOAD_SERVO);


    // Configure GPIO pins for motor control
    pinMode(F_PULSE, OUTPUT);
    pinMode(F_DIR, OUTPUT);
    pinMode(Y_PULSE, OUTPUT);
    pinMode(Y_DIR, OUTPUT);
    pinMode(Y_CLR, OUTPUT);
    digitalWrite(Y_CLR, LOW); // Ensure clear is inactive

    // Set up encoder pins
    pinMode(Fuji_EN_A, INPUT_PULLUP);
    pinMode(Fuji_EN_B, INPUT_PULLUP);
    pinMode(Fuji_EN_Z, INPUT_PULLUP);
   
    pinMode(Yas_EN_A, INPUT_PULLUP);
    pinMode(Yas_EN_B, INPUT_PULLUP);
    pinMode(Yas_EN_Z, INPUT_PULLUP);
   
       // Attach interrupts for encoders
    attachInterrupt(digitalPinToInterrupt(Fuji_EN_A), readFujiEncoder, CHANGE);
    attachInterrupt(digitalPinToInterrupt(Yas_EN_A), readYasEncoder, CHANGE);
}

void loop() {
    if (!isInitialized) {
        if (Serial.available()) {
            String input = Serial.readStringUntil('\n');
            input.trim();
            if (input == "55") {
                Serial.println("Starting MCP23017 setup...");

                //DACs Setup
                WireF.begin(F_DAC_SDA, F_DAC_SCL);  // For DAC_Ty an azimuth motor

                // Initialize I2C
                Wire.begin(SDA_I2C, SCL_I2C);
                Wire.setClock(200000); // Lower I2C speed for stability
                dac_Tf.begin(0x61, &WireF);  // Fuji DAC on custom bus

                if (!mcp.begin_I2C(0x20, &Wire)) {
                    Serial.println("MCP23017 initialization failed!");
                    while (1);
                }

                // Configure Outputs
                mcp.pinMode(8, OUTPUT);
                mcp.pinMode(9, OUTPUT);
                mcp.pinMode(10, OUTPUT);
                mcp.pinMode(11, OUTPUT);
                mcp.pinMode(7, OUTPUT);
                mcp.pinMode(6, OUTPUT);
                mcp.pinMode(5, OUTPUT);

                // Configure Inputs
                mcp.pinMode(4, INPUT_PULLDOWN);
                mcp.pinMode(12, INPUT);
                mcp.pinMode(13, INPUT);
                mcp.pinMode(14, INPUT);
                mcp.pinMode(15, INPUT);
                

                Serial.println("MCP23017 Initialized Successfully!");

                isInitialized = true;
                mcp.digitalWrite(8, HIGH); // Activate 2 Servo (After enter command ID 55)
                mcp.digitalWrite(9, LOW); // Initialize
                mcp.digitalWrite(10, LOW); // Initialize
                mcp.digitalWrite(11, LOW); // Initialize



            } else {
                Serial.println("Invalid command. Enter '55' to start MCP23017 setup.");
            }
        }
        return;
    }

    if (runStartTime > 0){
        esc1.writeMicroseconds(BLAST_SPEED);
        esc2.writeMicroseconds(BLAST_SPEED);
    } else {
        esc1.writeMicroseconds(0);
        esc2.writeMicroseconds(0);
    }

    if (Serial.available()) {
        String command = Serial.readStringUntil('\n');
        command.trim();
        handleCommand(command);
    }

    // Read Output signal
int fujiOK = mcp.digitalRead(12);
  int fujiFinishedInput = mcp.digitalRead(13); // Rename to avoid confusion
  int yaskawaOK = mcp.digitalRead(14);
  int yaskawaFinishedInput = mcp.digitalRead(15); // Rename to avoid confusion
  int home_sw = mcp.digitalRead(4);

  // Update the global finished variables based on the input pin states
  fujiFinished = fujiFinishedInput;
  yaskawaFinished = yaskawaFinishedInput;
}


void handleCommand(String command) {
    if (command == "00") {
        homeSetting();
    }
    else if (command == "0"){
        mcp.digitalWrite(8, LOW);
    }
    else if (command == "1"){
        mcp.digitalWrite(8,HIGH);
    }
    else if (command.startsWith("F") || command.startsWith("Y")) {
        if (command.length() >= 2) { // Corrected length check to >= 2 to allow single digit angles
            handleJog(command);
        } else {
            Serial.println("Error: Invalid angle format");
        }
    }
    else if ((command.startsWith("F") || command.startsWith("Y")) &&
             (command.endsWith("H") || command.endsWith("L"))) {
        setIO(command);

    } else if (command.startsWith("G")){
        Gun(command);

    } else if (command.startsWith("(") && command.endsWith(")")) {
        turn_to_target(command);

    } else {
        Serial.println("Error: Invalid command format");
        Serial.println("Valid formats:");
        Serial.println("  - Home: 00");
        Serial.println("  - Servo Power On: 1");
        Serial.println("  - Servo Power Off: 0");
        Serial.println("  - Motor Jog: F[angle] or Y[angle]");
        Serial.println("  - I/O Control: F[pin]H/F[pin]L or Y[pin]H/Y[pin]L (pin 1-3)");
        Serial.println("  - Shooting: GO1, GO2");
        Serial.println("  - Turn to Target: (x,y,z)");
    }
}

void homeSetting() {
    fujiPosition = 0;
    yasPosition = 0;
    delay(1000);

    Serial.println("Running home setting...");

    // Step 1: Set mode to speed mode and CW direction
    digitalWrite(Y_DIR, HIGH); //Set yaskawa direction (Using position control mode of altitude motor) moving down

    mcp.digitalWrite(9, HIGH);   // Speed mode ON (Fuji)
    mcp.digitalWrite(10, LOW);   // CCW OFF (Fuji)
    mcp.digitalWrite(11, HIGH);  // CW ON (Fuji)
    v = 6;

    // Step 2: Move until switch is hit 
    while (!isSwitchPressedStable(4)) {
            zero_orderhold_F(v);
            delay(10);
        }
    // Step 3: Stop and zero position
    zero_orderhold_F(0.0);
    delay(100);

    N_fujiPosition = abs(fujiPosition);
    fujiPosition = 0;
    yasPosition = 0;

    // Step 4: Iterative homing refinement
    int iteration = 0;
    while (abs(N_fujiPosition - fujiPosition) && iteration < 10 ) {
        Serial.print("Iteration: "); Serial.println(iteration);
        v = v / 1.2;

        int target_f = -0.1 * N_fujiPosition; //Avoid divide by 0
        Serial.print("target_f: "); Serial.println(target_f);

        // Move back (CCW)
        mcp.digitalWrite(10, HIGH);  // CCW ON (Azimuth)
        mcp.digitalWrite(11, LOW);   // CW OFF

        //Not sure about this loop also!!
        while (abs(target_f - fujiPosition) > 1) { //Do both condition (need to finish both before exit the loop)
            int error = target_f - fujiPosition;
            float k = constrain(abs(error / float(target_f)), 0.1, 1.0);
            zero_orderhold_F(k * v);
            delay(1);
        }

        zero_orderhold_F(0.0);
        delay(100);

        // Move forward (CW) to switch again
        mcp.digitalWrite(11, HIGH);  // CW ON
        mcp.digitalWrite(10, LOW);   // CCW OFF
        while (!isSwitchPressedStable(4)) {
            zero_orderhold_F(v);
            delay(10);
        }

        zero_orderhold_F(0.0);
        delay(100);
        N_fujiPosition = abs(fujiPosition);
        fujiPosition = 0;
        iteration++;
    }

    // Step 5: Move backward to offset position
    int target_f = -5220;
    mcp.digitalWrite(10, HIGH);  // CCW ON
    mcp.digitalWrite(11, LOW);   // CW OFF

    while (abs(target_f - fujiPosition) > 1) {
        v = 4;
        int error = target_f - fujiPosition;
        float k = constrain(abs(error / float(target_f)), 0.1, 1.0);
        zero_orderhold_F(k * v);
        delay(1);
    }

    // Step 6: Stop motor, cleanup, and reset position
    zero_orderhold_F(0.0);
    fujiPosition = 0;

    mcp.digitalWrite(9, LOW);  // Speed mode OFF
    mcp.digitalWrite(10, LOW);
    mcp.digitalWrite(11, LOW);

    Serial.println("Fuji Homing complete.");


    //Now working at the altitude motor, With have limit switch connected to GPA3
    while (!isSwitchPressedStable(3)) {
        generatePulses(Y_PULSE, 100, 5);
    }
    yasPosition = 0;

    int target_y = -2157;

    while (target_y - yasPosition < 1) {
        digitalWrite(Y_DIR, LOW);
        generatePulses(Y_PULSE, 1, 30);
    }

    Serial.println(yasPosition);

    yasPosition = 0;
    delay(10);
    digitalWrite(Y_CLR, HIGH); // Ensure clear is inactive
    delay(200);
    digitalWrite(Y_CLR, LOW);

    delay(50);
    handleJog("Y-3");
    yasPosition = 0;
    Serial.println("Yaskawa Homing complete.");
}

void setIO(String command) { //These function setIO of the servo drive
    // Validate command length
    if (command.length() < 3) {
        Serial.println("Error: Command too short");
        return;
    }
    // Extract pin number (middle characters)
    int pin = command.substring(1, command.length() - 1).toInt();
    String lastDigitStr = command.substring(command.length() - 1);
    // Validate pin number range
    if (pin < 1 || pin > 3) {
        Serial.println("Error: Invalid pin number (1-7)");
        return;
    }
    // Process command
    if (command.startsWith("F")) {
        if (command.endsWith("H")) {
            mcp.digitalWrite(pin + 8, HIGH);  
            Serial.print("Set Fuji DI");
        } else if (command.endsWith("L")) {
            mcp.digitalWrite(pin + 8, LOW);
            Serial.print("Clear Fuji DI");
        }
    } 
    else if (command.startsWith("Y")) {
        if (command.endsWith("H")) {
            mcp.digitalWrite(8 - pin, HIGH);  
            Serial.print("Set Yaskawa DI");
        } else if (command.endsWith("L")) {
            mcp.digitalWrite(8 - pin, LOW);
            Serial.print("Clear Yaskawa DI");
        }
    }
    
    Serial.print(pin);
    Serial.print(" to ");
    Serial.println(lastDigitStr);
}

void handleJog(String command) {
    //Chk fuji position
    float F_present = fujiPosition * F_deg_per_pulse ; //Define present angle of Azimuth motor

    //Chk Yas position
    float Y_present = yasPosition * Y_deg_per_pulse ; //Define present angle of Altitude motor

    float angle = command.substring(1, command.length()).toFloat(); //Angle define in real number

    //Fuji motor 4096 encoder pulse for 1 rotation
    if (command.startsWith("F")) {
  //      Serial.println("Rotating Fuji motor to" + String(angle) + " degrees.");
        float e_angle = angle - F_present; //Define how much to move
        float pulseCount = abs(e_angle)*72.84736*5; // Calculate the number of pulses (Azimuth motor)
        pulseCount = round(pulseCount);
        int speed_microseconds = round(abs(e_angle) * (-2/90.0) + 10);

        if (e_angle < 0) {
            digitalWrite(F_DIR, LOW); // Set direction for negative rotation
        } else {
            digitalWrite(F_DIR, HIGH); // Set direction for positive rotation
            delay(2);
        }

        generatePulses(F_PULSE, pulseCount, speed_microseconds);
    
    }

    //Yaskawa encoder pulse 4096 pulse per rotation
    if (command.startsWith("Y")) {
  //      Serial.println("Rotating Yaskawa motor " + String(angle) + " degrees.");
        float e_angle = angle - Y_present; //Define how much to move
        float pulseCount = abs(8 * e_angle * 99.695); // Calculate the number of pulses (Azimuth motor)
        pulseCount = round(pulseCount);

        if (e_angle < 0) {
            digitalWrite(Y_DIR, HIGH); // Set direction for negative rotation
        } else {
            digitalWrite(Y_DIR, LOW); // Set direction for positive rotation
            delay(2);
        }
        generatePulses(Y_PULSE, pulseCount, 2); // Generate pulses for Yaskawa motor
        delay(10);
        digitalWrite(Y_CLR, HIGH); // Ensure clear is inactive
        delay(200);
        digitalWrite(Y_CLR, LOW); 
    }
}

void jogAzimuthTask(void *pvParameters) {
    MotorControlData *data = (MotorControlData *)pvParameters; 
    float targetAngle = data->targetAngle;
    
    float F_present = fujiPosition * F_deg_per_pulse;
    float e_angle = targetAngle - F_present;
    
    long pulseCount = abs(e_angle) * 72.84736 * 5; 
    pulseCount = round(pulseCount);

    int speed_microseconds = (int)(abs(e_angle) * (-2.0/90.0) + 2.0);
    // --- IMPORTANT: ADD THESE CONSTRAINTS BACK ---
    if (speed_microseconds < 1) speed_microseconds = 2; 
    if (speed_microseconds > 100) speed_microseconds = 100;
    // ---------------------------------------------

    if (e_angle < 0) {
        digitalWrite(F_DIR, LOW); 
    } else {
        digitalWrite(F_DIR, HIGH); 
        // --- RECOMMENDED: UNCOMMENT THIS ---
        vTaskDelay(pdMS_TO_TICKS(1)); // Short delay to allow direction pin to stabilize
        // ------------------------------------
    }

    const int BATCH_SIZE = 50; 
    for (long i = 0; i < pulseCount; i++) {
        digitalWrite(F_PULSE, HIGH);
        delayMicroseconds(speed_microseconds); 
        digitalWrite(F_PULSE, LOW);
        delayMicroseconds(speed_microseconds);

        if ((i + 1) % BATCH_SIZE == 0) {
            vTaskDelay(1); // Yield to other tasks
        }
    }
    
    azimuthTaskHandle = NULL;
    vTaskDelete(NULL); 
}
  
void jogAltitudeTask(void *pvParameters) {
    MotorControlData *data = (MotorControlData *)pvParameters;
    float targetAngle = data->targetAngle;
    
    float Y_present = yasPosition * Y_deg_per_pulse; 
    float e_angle = targetAngle - Y_present;
    
    long pulseCount = abs(8 * e_angle * 99.695); 
    pulseCount = round(pulseCount);

    if (e_angle < 0) {
        digitalWrite(Y_DIR, HIGH); 
    } else {
        digitalWrite(Y_DIR, LOW); 
        // --- RECOMMENDED: UNCOMMENT THIS ---
        vTaskDelay(pdMS_TO_TICKS(1)); // Short delay to allow direction pin to stabilize
        // ------------------------------------
    }

    const int BATCH_SIZE = 50; 
    const int PULSE_DELAY_US = 2; // Fixed pulse delay for altitude motor
    for (long i = 0; i < pulseCount; i++) {
        digitalWrite(Y_PULSE, HIGH);
        delayMicroseconds(PULSE_DELAY_US); 
        digitalWrite(Y_PULSE, LOW);
        delayMicroseconds(PULSE_DELAY_US);

        if ((i + 1) % BATCH_SIZE == 0) {
            vTaskDelay(1); // Yield to other tasks
        }
    }
    
    vTaskDelay(pdMS_TO_TICKS(10));
    digitalWrite(Y_CLR, HIGH); 
    vTaskDelay(pdMS_TO_TICKS(200)); 
    digitalWrite(Y_CLR, LOW); 

    altitudeTaskHandle = NULL; 
    vTaskDelete(NULL); 
}

void shotSchedulerTask(void *pvParameters) {
    ShotSchedulerData *data = (ShotSchedulerData *)pvParameters;
    int delayMs = data->delayBeforeShootMs;

    // Perform the non-blocking delay. This yields CPU, allowing motors to turn.
    if (delayMs > 0) {
        vTaskDelay(pdMS_TO_TICKS(delayMs)); 
    }

    servo.write(120);
    vTaskDelay(pdMS_TO_TICKS(200)); 
    servo.write(0);
    vTaskDelay(pdMS_TO_TICKS(200)); 
    servo.write(120);
    vTaskDelay(pdMS_TO_TICKS(400));
    vTaskDelete(NULL);
}

void Gun(String command) {
    if (command == "G01") {
      Serial.println("\n========= ESC CALIBRATION MODE =========");
      waitForEnter("1) Disconnect ESC power, then press Enter.");
      Serial.println("Sending MAX throttle signal...");
      esc1.writeMicroseconds(MAX_PULSE_WIDTH);
      esc2.writeMicroseconds(MAX_PULSE_WIDTH);
      waitForEnter("2) Reconnect ESC power and wait for beeps, then press Enter.");
      Serial.println("Setting to MIN throttle...");
      esc1.writeMicroseconds(MIN_PULSE_WIDTH);
      esc2.writeMicroseconds(MIN_PULSE_WIDTH);
      delay(3000);
      Serial.println("Calibration completed!");

    } else if (command == "G20") {
      Serial.println("Starting ESC...");

      int i = 0;

      while (i < BLAST_SPEED){
        esc1.writeMicroseconds(i);
        esc2.writeMicroseconds(i);
        i = i + 2;
        delay(2);
      }

        runStartTime = 1;      
      
    } else if (command == "G21") {
      Serial.println("Stopping ESC...");
    runStartTime = 0;

    } else if (command == "G30"){
        runStartTime = 1;

    } else if (command == "G22"){
        servo.write(120);
        delay(200); // Non-blocking delay for 200ms - FIX IS HERE
        servo.write(0);
        delay(200); // Non-blocking delay for 200ms - FIX IS HERE
        servo.write(120); // Non-blocking delay for 400ms - FIX IS HERE
    
    } else if (command == "G00"){
        runStartTime = 0;
        handleJog("F000"); // Return Azimuth to 0
        handleJog("Y000"); // Return Altitude to 0

    } else {
      Serial.println("Error: Invalid Gun command.");
    }
  }

void turn_to_target(String command) {
    //Now My turn to target got another problem, It not turn but only Shoot, Shoot ids good it operate at appropriate time but turn is not.
    float x, y, z, t_raw; // Use t_raw for parsed value
    // Attempt to parse the (x,y,z,t) coordinates from the command string
    if (sscanf(command.c_str(), "(%f,%f,%f,%f)", &x, &y, &z, &t_raw) == 4) {
        // Apply your specific offsets
        x = -(x) + 17.5 + 151.5; 
        y = -(y) + 2; 
        //y = y + 2;
        //x = x + 17.5;
 
        // Basic validation for coordinates (adjust as per your system's valid range)
        if (x >= 0 && z >= 0) {

            float azimuth_angle_rad = atan2(y, x);
            float r = sqrt(pow(x, 2) + pow(y, 2)); // Horizontal distance
            float d = sqrt(pow(r, 2) + pow(z, 2));
            float altitude_angle_rad = atan2((z - 35.5), r); // Adjust 35.5 based on your setup's height
            int shoot_delay_ms = round(abs(t_raw) - 120 - (d*1000/2200)); // Assuming 't' is already in ms. If it's a scaled value, use your scaling.
            shotSchedulerGlobalData.delayBeforeShootMs = shoot_delay_ms;
            xTaskCreate( shotSchedulerTask, "ShotScheduler", 2048, &shotSchedulerGlobalData, SHOOT_TASK_PRIORITY, &shotSchedulerTaskHandle);

            // Calculate azimuth and altitude angles in radians
        

            // Apply azimuth compensation 
            if (azimuth_angle_rad > 0) {
                compensate_Azi = (0.00013) * r;
            } else {
                compensate_Azi = (0.000003) * r; 
            }
            azimuth_angle_rad = azimuth_angle_rad + compensate_Azi * azimuth_angle_rad;

            // Convert angles to degrees
            float azimuth_angle_deg = azimuth_angle_rad * 180.0 / PI;
            float altitude_angle_deg = altitude_angle_rad * 180.0 / PI;

            //String FtargetCommand = "F" + String(azimuth_angle_deg);
            String YtargetCommand = "Y" + String(altitude_angle_deg);

            MotorControlData azimuthData;
            azimuthData.targetAngle = azimuth_angle_deg;
            xTaskCreate(jogAzimuthTask, "AzimuthJog", 2048, &azimuthData, AZIMUTH_TASK_PRIORITY, &azimuthTaskHandle);


            //xTaskCreate(jogAzimuthTask, "AzimuthJog", STACK_SIZE, &azimuthGlobalData, AZIMUTH_TASK_PRIORITY, &azimuthTaskHandle);
            //xTaskCreate(jogAltitudeTask, "AltitudeJog", STACK_SIZE, &altitudeGlobalData, ALTITUDE_TASK_PRIORITY, &altitudeTaskHandle);
            handleJog(YtargetCommand);


        } else {
            Serial.println("Error: Invalid target coordinate (x or z negative or out of bounds).");
        }
    } else {
        Serial.println("Error: Invalid target coordinate format. Expected (x,y,z,t)");
    }
}
