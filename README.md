# ConvoSense 🚧

### Intelligent Conveyor Belt Health Monitoring & Predictive Maintenance System

**SIH 2026 • Problem Statement: SIH26008**

ConvoSense is a low-cost intelligent monitoring system for **iron ore mining conveyor belts**. It monitors vibration, RPM, motor current, and load to detect abnormal conditions, evaluate conveyor health, assess risk, and generate real-time alerts.

---

## 🔍 Problem

Conveyor belt failures can cause:

* Unplanned downtime
* Production losses
* High maintenance costs
* Equipment damage
* Safety risks

Traditional maintenance may identify problems only after significant degradation.

---

## 💡 Solution

ConvoSense uses **ESP32-based sensing + Python analytics + real-time visualization** to continuously monitor conveyor operating conditions.

**Core workflow:**

```text
Sensors
   ↓
ESP32
   ↓
Data Processing
   ↓
Anomaly Detection
   ↓
Health & Risk Assessment
   ↓
Streamlit Dashboard
   ↓
Alerts
```

---

## 📡 Parameters Monitored

| Parameter     | Purpose                               |
| ------------- | ------------------------------------- |
| Vibration     | Detect abnormal mechanical behavior   |
| RPM           | Identify speed variations             |
| Motor Current | Detect abnormal motor loading         |
| Load          | Monitor conveyor loading and overload |

---

## 🧠 Technology Stack

**Hardware**

* ESP32
* Vibration Sensor
* RPM Sensor
* Current Sensor
* Load Cell

**Programming & Processing**

* Python
* C/C++
* Arduino IDE
* NumPy
* Pandas

**Analytics**

* Threshold Analysis
* Anomaly Detection
* Machine Learning
* Sensor Fusion

**Dashboard**

* Streamlit
* Plotly

**Communication**

* Wi-Fi
* MQTT

---

## 📊 Key Features

* Real-time parameter monitoring
* Abnormal-condition detection
* Conveyor Health Score
* Risk-level assessment
* Sensor trend visualization
* Intelligent alerts
* Low-cost architecture
* Retrofit-friendly design

---

## 🖥️ Dashboard

The Streamlit dashboard provides:

```text
┌──────────────────────────────────┐
│       CONVEYOR HEALTH            │
├──────────┬──────────┬────────────┤
│Vibration │   RPM    │   Current  │
├──────────┴──────────┴────────────┤
│              Load                │
├──────────────────────────────────┤
│        Health Score              │
│        Risk Level                │
├──────────────────────────────────┤
│     Trends & Alerts              │
└──────────────────────────────────┘
```

---

## 🚀 Development Status

| Component                   | Status    |
| --------------------------- | --------- |
| ESP32 Architecture          | ✅         |
| Sensor Data Acquisition     | ✅         |
| Data Simulation             | ✅         |
| Streamlit Dashboard         | ✅         |
| Threshold Monitoring        | ✅         |
| Health Assessment           | ✅         |
| Risk Classification         | ✅         |
| Anomaly Detection           | ✅         |
| Physical Conveyor Prototype | 🔄 Future |
| Industrial Field Validation | 🔄 Future |

---

## 🔮 Future Scope

* Industrial-grade sensors
* Computer vision for belt damage detection
* Advanced ML-based failure prediction
* Edge AI
* Cloud monitoring
* SCADA/PLC integration
* Long-term field-data validation
* Industrial deployment

---

## 👥 Team ConvoSense

**Smart Monitoring • Early Detection • Predictive Maintenance**

Developed for **Smart India Hackathon 2026**.

---

> **From reactive maintenance to intelligent conveyor health management.**
