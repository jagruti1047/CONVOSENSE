
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Conveyor Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Intelligent Conveyor Belt Health & Predictive Maintenance")

st.write(
    "Simulation-based monitoring of conveyor operating parameters "
    "using multi-sensor analysis, vibration signal processing, "
    "rule-based sensor fusion and failure-risk estimation."
)


# =========================================================
# REFERENCE VALUES
# =========================================================

NORMAL_VIBRATION = 3.0
WARNING_VIBRATION = 6.0

NORMAL_RPM = 1000.0
NORMAL_CURRENT = 10.0

MAX_LOAD = 100.0


# =========================================================
# SCENARIO SELECTION
# =========================================================

st.header("🎯 Conveyor Operating Scenario")

scenario = st.selectbox(
    "Select a simulated operating condition",
    [
        "Normal Operation",
        "Belt Joint Deterioration",
        "Belt Damage",
        "Belt Slip",
        "Overload",
        "Misalignment"
    ]
)


# =========================================================
# BASE SENSOR VALUES
# =========================================================

scenario_values = {

    "Normal Operation": {
        "vibration": 2.0,
        "rpm": 1000.0,
        "current": 8.0,
        "load": 50.0
    },

    "Belt Joint Deterioration": {
        "vibration": 5.0,
        "rpm": 990.0,
        "current": 9.5,
        "load": 60.0
    },

    "Belt Damage": {
        "vibration": 7.5,
        "rpm": 980.0,
        "current": 11.0,
        "load": 65.0
    },

    "Belt Slip": {
        "vibration": 4.0,
        "rpm": 850.0,
        "current": 12.0,
        "load": 60.0
    },

    "Overload": {
        "vibration": 5.0,
        "rpm": 950.0,
        "current": 17.0,
        "load": 95.0
    },

    "Misalignment": {
        "vibration": 6.5,
        "rpm": 970.0,
        "current": 13.0,
        "load": 70.0
    }
}


selected = scenario_values[scenario]


# =========================================================
# INPUT PARAMETERS
# =========================================================

st.header("🔧 Conveyor Sensor Inputs")

col1, col2 = st.columns(2)


with col1:

    vibration = st.number_input(
        "Vibration (m/s²)",
        min_value=0.0,
        max_value=20.0,
        value=float(selected["vibration"]),
        step=0.1
    )

    rpm = st.number_input(
        "Motor / Belt RPM",
        min_value=0.0,
        max_value=2000.0,
        value=float(selected["rpm"]),
        step=10.0
    )


with col2:

    current = st.number_input(
        "Motor Current (A)",
        min_value=0.0,
        max_value=50.0,
        value=float(selected["current"]),
        step=0.1
    )

    load = st.number_input(
        "Load Cell (kg)",
        min_value=0.0,
        max_value=100.0,
        value=float(selected["load"]),
        step=0.5
    )


# =========================================================
# RUN BUTTON
# =========================================================

run_simulation = st.button(
    "▶ RUN MONITORING & PREDICTION",
    type="primary",
    use_container_width=True
)


if run_simulation:

    # =====================================================
    # TIME / SIGNAL SETTINGS
    # =====================================================

    duration = 10
    sample_rate = 100

    t = np.linspace(
        0,
        duration,
        duration * sample_rate,
        endpoint=False
    )

    np.random.seed(42)


    # =====================================================
    # FAULT-SPECIFIC VIBRATION FREQUENCY
    # =====================================================

    if scenario == "Normal Operation":

        vibration_frequency = 5

    elif scenario == "Belt Joint Deterioration":

        vibration_frequency = 8

    elif scenario == "Belt Damage":

        vibration_frequency = 12

    elif scenario == "Belt Slip":

        vibration_frequency = 4

    elif scenario == "Overload":

        vibration_frequency = 6

    elif scenario == "Misalignment":

        vibration_frequency = 10

    else:

        vibration_frequency = 5


    # =====================================================
    # 1. SIMULATED VIBRATION SIGNAL
    # =====================================================

    vibration_signal = (

        vibration

        + vibration * 0.10
        * np.sin(
            2 * np.pi * vibration_frequency * t
        )

        + np.random.normal(
            0,
            max(vibration * 0.02, 0.01),
            len(t)
        )
    )


    # Add additional vibration components for damage
    if scenario == "Belt Joint Deterioration":

        vibration_signal += (
            vibration * 0.15
            * np.sin(2 * np.pi * 16 * t)
        )


    elif scenario == "Belt Damage":

        vibration_signal += (
            vibration * 0.20
            * np.sin(2 * np.pi * 20 * t)
        )


    elif scenario == "Misalignment":

        vibration_signal += (
            vibration * 0.18
            * np.sin(2 * np.pi * 15 * t)
        )


    # =====================================================
    # 2. RPM SIGNAL
    # =====================================================

    rpm_signal = (

        rpm

        + 5
        * np.sin(
            2 * np.pi * 0.5 * t
        )

        + np.random.normal(
            0,
            1,
            len(t)
        )
    )


    # =====================================================
    # 3. MOTOR CURRENT SIGNAL
    # =====================================================

    current_signal = (

        current

        + current * 0.05
        * np.sin(
            2 * np.pi * 1 * t
        )

        + np.random.normal(
            0,
            max(current * 0.01, 0.01),
            len(t)
        )
    )


    # =====================================================
    # 4. LOAD SIGNAL
    # =====================================================

    load_signal = (

        load

        + load * 0.03
        * np.sin(
            2 * np.pi * 0.2 * t
        )

        + np.random.normal(
            0,
            max(load * 0.005, 0.01),
            len(t)
        )
    )


    # =====================================================
    # RMS VIBRATION ANALYSIS
    # =====================================================

    vibration_rms = np.sqrt(
        np.mean(
            vibration_signal ** 2
        )
    )

    vibration_rms = round(
        vibration_rms,
        2
    )


    # =====================================================
    # FFT ANALYSIS
    # =====================================================

    n = len(vibration_signal)

    fft_values = np.fft.fft(
        vibration_signal
    )

    fft_frequencies = np.fft.fftfreq(
        n,
        1 / sample_rate
    )

    positive_mask = (
        fft_frequencies >= 0
    )

    positive_frequencies = (
        fft_frequencies[
            positive_mask
        ]
    )

    positive_fft = np.abs(
        fft_values[
            positive_mask
        ]
    ) / n


    # Remove DC component
    positive_frequencies = positive_frequencies[1:]
    positive_fft = positive_fft[1:]


    # Dominant frequency
    if len(positive_fft) > 0:

        dominant_index = np.argmax(
            positive_fft
        )

        dominant_frequency = round(
            positive_frequencies[
                dominant_index
            ],
            2
        )

    else:

        dominant_frequency = 0


    # =====================================================
    # 5. VIBRATION CONDITION
    # =====================================================

    if vibration <= NORMAL_VIBRATION:

        vibration_condition = "NORMAL"
        vibration_risk = 0

    elif vibration <= WARNING_VIBRATION:

        vibration_condition = "WARNING"
        vibration_risk = 20

    else:

        vibration_condition = "CRITICAL"
        vibration_risk = 40


    # =====================================================
    # 6. RPM ANALYSIS
    # =====================================================

    rpm_deviation = (
        abs(rpm - NORMAL_RPM)
        / NORMAL_RPM
    ) * 100


    if rpm_deviation <= 5:

        rpm_condition = "NORMAL"
        rpm_risk = 0

    elif rpm_deviation <= 10:

        rpm_condition = "WARNING"
        rpm_risk = 15

    else:

        rpm_condition = "CRITICAL"
        rpm_risk = 25


    # =====================================================
    # 7. MOTOR CURRENT ANALYSIS
    # =====================================================

    if current <= NORMAL_CURRENT:

        current_condition = "NORMAL"
        current_risk = 0

    elif current <= 15:

        current_condition = "WARNING"
        current_risk = 15

    else:

        current_condition = "CRITICAL"
        current_risk = 25


    # =====================================================
    # 8. LOAD ANALYSIS
    # =====================================================

    load_percentage = (
        load / MAX_LOAD
    ) * 100


    if load_percentage <= 70:

        load_condition = "NORMAL"
        load_risk = 0

    elif load_percentage <= 90:

        load_condition = "WARNING"
        load_risk = 10

    else:

        load_condition = "CRITICAL"
        load_risk = 20


    # =====================================================
    # 9. RULE-BASED SENSOR FUSION
    # =====================================================

    risk_score = (

        vibration_risk
        + rpm_risk
        + current_risk
        + load_risk

    )

    risk_score = min(
        risk_score,
        100
    )


    # =====================================================
    # 10. HEALTH SCORE
    # =====================================================

    health_score = (
        100 - risk_score
    )


    # =====================================================
    # 11. SENSOR SEVERITY
    # =====================================================

    # Vibration severity

    vibration_severity = min(
        (vibration / WARNING_VIBRATION)
        * 100,
        100
    )


    # RPM severity

    rpm_severity = min(
        (rpm_deviation / 10)
        * 100,
        100
    )


    # Current severity

    if current <= NORMAL_CURRENT:

        current_severity = 0

    else:

        current_severity = min(
            (
                (current - NORMAL_CURRENT)
                / 5
            ) * 100,
            100
        )


    # Load severity

    if load_percentage <= 70:

        load_severity = 0

    else:

        load_severity = min(
            (
                (load_percentage - 70)
                / 30
            ) * 100,
            100
        )


    # =====================================================
    # 12. FAILURE-RISK ESTIMATION
    # =====================================================

    prediction_risk = (

        0.40 * vibration_severity

        + 0.25 * rpm_severity

        + 0.20 * current_severity

        + 0.15 * load_severity

    )


    prediction_risk = min(
        round(prediction_risk),
        100
    )


    # =====================================================
    # 13. FAILURE-RISK LEVEL
    # =====================================================

    if prediction_risk < 25:

        prediction_level = "LOW"

        prediction_message = (
            "Conveyor is operating normally. "
            "Continue routine monitoring."
        )

        prediction_horizon = (
            "No immediate failure indication"
        )


    elif prediction_risk < 50:

        prediction_level = "MODERATE"

        prediction_message = (
            "Early abnormal behavior detected. "
            "Increase monitoring frequency."
        )

        prediction_horizon = (
            "Monitor over upcoming operating cycles"
        )


    elif prediction_risk < 75:

        prediction_level = "HIGH"

        prediction_message = (
            "Elevated failure risk detected. "
            "Preventive inspection should be scheduled."
        )

        prediction_horizon = (
            "Possible degradation developing"
        )


    else:

        prediction_level = "CRITICAL"

        prediction_message = (
            "Very high failure risk detected. "
            "Immediate conveyor inspection is recommended."
        )

        prediction_horizon = (
            "Possible near-term failure"
        )


    # =====================================================
    # 14. CURRENT CONVEYOR CONDITION
    # =====================================================

    if health_score >= 80:

        condition = "NORMAL"
        status = "🟢"

    elif health_score >= 60:

        condition = "WARNING"
        status = "🟡"

    else:

        condition = "CRITICAL"
        status = "🔴"


    # =====================================================
    # OUTPUT
    # =====================================================

    st.divider()

    st.header("📊 Monitoring Results")


    # =====================================================
    # TOP METRICS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Health Score",
            f"{health_score}/100"
        )


    with col2:

        st.metric(
            "Current Condition",
            f"{status} {condition}"
        )


    with col3:

        st.metric(
            "Failure Risk",
            f"{prediction_risk}%"
        )


    with col4:

        st.metric(
            "Load Utilization",
            f"{load_percentage:.1f}%"
        )


    # =====================================================
    # SCENARIO
    # =====================================================

    st.subheader("🎯 Simulated Fault Scenario")

    st.info(
        f"Current simulation: **{scenario}**"
    )


    # =====================================================
    # FAILURE-RISK ASSESSMENT
    # =====================================================

    st.subheader("🔮 Failure-Risk Assessment")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Risk",
            f"{prediction_risk}%"
        )


    with col2:

        st.metric(
            "Risk Level",
            prediction_level
        )


    with col3:

        st.metric(
            "Assessment",
            prediction_horizon
        )


    if prediction_level == "LOW":

        st.success(
            "🟢 " + prediction_message
        )


    elif prediction_level == "MODERATE":

        st.info(
            "🔵 " + prediction_message
        )


    elif prediction_level == "HIGH":

        st.warning(
            "🟡 " + prediction_message
        )


    else:

        st.error(
            "🔴 " + prediction_message
        )


    # =====================================================
    # VIBRATION SIGNAL ANALYSIS
    # =====================================================

    st.divider()

    st.header("📈 Vibration Signal Analysis")


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Vibration Input",
            f"{vibration:.2f} m/s²"
        )


    with col2:

        st.metric(
            "Vibration RMS",
            f"{vibration_rms:.2f} m/s²"
        )


    st.write(
        f"**Dominant vibration frequency:** "
        f"{dominant_frequency:.2f} Hz"
    )


    # =====================================================
    # VIBRATION TIME-DOMAIN GRAPH
    # =====================================================

    fig1, ax1 = plt.subplots(
        figsize=(10, 4)
    )


    ax1.plot(
        t,
        vibration_signal,
        label="Simulated Vibration"
    )


    ax1.axhline(
        NORMAL_VIBRATION,
        linestyle="--",
        label="Normal Limit"
    )


    ax1.axhline(
        WARNING_VIBRATION,
        linestyle="--",
        label="Warning Limit"
    )


    ax1.set_title(
        "Vibration Signal — Accelerometer"
    )


    ax1.set_xlabel(
        "Time (s)"
    )


    ax1.set_ylabel(
        "Acceleration (m/s²)"
    )


    ax1.legend()

    ax1.grid(True)

    st.pyplot(fig1)


    # =====================================================
    # FFT GRAPH
    # =====================================================

    st.subheader("🔬 FFT Frequency Analysis")


    fig_fft, ax_fft = plt.subplots(
        figsize=(10, 4)
    )


    ax_fft.plot(
        positive_frequencies,
        positive_fft
    )


    ax_fft.set_xlim(
        0,
        50
    )


    ax_fft.set_title(
        "Vibration Frequency Spectrum — FFT"
    )


    ax_fft.set_xlabel(
        "Frequency (Hz)"
    )


    ax_fft.set_ylabel(
        "Amplitude"
    )


    ax_fft.grid(True)

    st.pyplot(fig_fft)


    st.caption(
        "FFT is used in the simulation to identify dominant "
        "frequency components in the vibration signal."
    )


    # =====================================================
    # SENSOR ANALYSIS TABLE
    # =====================================================

    st.divider()

    st.header("🔍 Multi-Sensor Analysis")


    data = {

        "Parameter": [

            "Vibration",
            "Vibration RMS",
            "Dominant Frequency",
            "RPM",
            "Motor Current",
            "Load"

        ],

        "Input": [

            f"{vibration:.2f} m/s²",
            f"{vibration_rms:.2f} m/s²",
            f"{dominant_frequency:.2f} Hz",
            f"{rpm:.0f} RPM",
            f"{current:.2f} A",
            f"{load:.2f} kg"

        ],

        "Condition": [

            vibration_condition,
            vibration_condition,
            "Signal Analysis",
            rpm_condition,
            current_condition,
            load_condition

        ],

        "Risk Contribution": [

            vibration_risk,
            "-",
            "-",
            rpm_risk,
            current_risk,
            load_risk

        ]

    }


    df = pd.DataFrame(
        data
    )


    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # FAULT DETECTION
    # =====================================================

    st.header("🚨 Fault Detection")


    faults = []


    # High vibration

    if vibration > WARNING_VIBRATION:

        faults.append(
            "High vibration — possible belt joint "
            "deterioration, belt damage or mechanical abnormality."
        )


    # RPM deviation

    if rpm_deviation > 10:

        faults.append(
            "Abnormal RPM — possible belt slip "
            "or drive problem."
        )


    # High current

    if current > 15:

        faults.append(
            "High motor current — possible overload "
            "or mechanical resistance."
        )


    # Excessive load

    if load_percentage > 90:

        faults.append(
            "Excessive conveyor load detected."
        )


    # Scenario-specific diagnosis

    if scenario == "Belt Joint Deterioration":

        faults.append(
            "Belt joint deterioration scenario — "
            "increased vibration indicates possible splice degradation."
        )


    elif scenario == "Belt Damage":

        faults.append(
            "Belt damage scenario — elevated vibration "
            "indicates possible belt structural damage."
        )


    elif scenario == "Belt Slip":

        faults.append(
            "Belt slip scenario — significant RPM deviation detected."
        )


    elif scenario == "Overload":

        faults.append(
            "Overload scenario — high load/current combination detected."
        )


    elif scenario == "Misalignment":

        faults.append(
            "Misalignment scenario — elevated vibration and "
            "RPM deviation indicate abnormal belt tracking."
        )


    # =====================================================
    # DISPLAY FAULTS
    # =====================================================

    if len(faults) == 0:

        st.success(
            "✅ No significant abnormality detected."
        )

    else:

        # Remove duplicate messages
        faults = list(
            dict.fromkeys(faults)
        )

        for fault in faults:

            st.warning(
                "⚠️ " + fault
            )


    # =====================================================
    # SENSOR TREND GRAPHS
    # =====================================================

    st.divider()

    st.header("📊 Sensor Trends")


    # =====================================================
    # RPM GRAPH
    # =====================================================

    fig2, ax2 = plt.subplots(
        figsize=(10, 3)
    )


    ax2.plot(
        t,
        rpm_signal
    )


    ax2.axhline(
        NORMAL_RPM,
        linestyle="--",
        label="Reference RPM"
    )


    ax2.set_title(
        "RPM Signal"
    )


    ax2.set_xlabel(
        "Time (s)"
    )


    ax2.set_ylabel(
        "RPM"
    )


    ax2.legend()

    ax2.grid(True)

    st.pyplot(fig2)


    # =====================================================
    # CURRENT GRAPH
    # =====================================================

    fig3, ax3 = plt.subplots(
        figsize=(10, 3)
    )


    ax3.plot(
        t,
        current_signal
    )


    ax3.axhline(
        NORMAL_CURRENT,
        linestyle="--",
        label="Normal Current Limit"
    )


    ax3.set_title(
        "Motor Current Signal"
    )


    ax3.set_xlabel(
        "Time (s)"
    )


    ax3.set_ylabel(
        "Current (A)"
    )


    ax3.legend()

    ax3.grid(True)

    st.pyplot(fig3)


    # =====================================================
    # LOAD GRAPH
    # =====================================================

    fig4, ax4 = plt.subplots(
        figsize=(10, 3)
    )


    ax4.plot(
        t,
        load_signal
    )


    ax4.axhline(
        MAX_LOAD,
        linestyle="--",
        label="Maximum Load"
    )


    ax4.set_title(
        "Load Cell Signal"
    )


    ax4.set_xlabel(
        "Time (s)"
    )


    ax4.set_ylabel(
        "Load (kg)"
    )


    ax4.legend()

    ax4.grid(True)

    st.pyplot(fig4)


    # =====================================================
    # MAINTENANCE RECOMMENDATION
    # =====================================================

    st.divider()

    st.header("🛠️ Maintenance Recommendation")


    if scenario == "Normal Operation":

        recommendation = (
            "Continue routine monitoring and scheduled maintenance."
        )


    elif scenario == "Belt Joint Deterioration":

        recommendation = (
            "Inspect belt joint/splice condition. "
            "Check for wear, cracks, loosening and progressive vibration."
        )


    elif scenario == "Belt Damage":

        recommendation = (
            "Inspect the belt for cracks, tears, edge damage and "
            "rubber deterioration. Schedule preventive maintenance."
        )


    elif scenario == "Belt Slip":

        recommendation = (
            "Inspect belt tension, drive pulley condition and "
            "belt-drive system for slip."
        )


    elif scenario == "Overload":

        recommendation = (
            "Reduce excessive loading and inspect motor, drive and "
            "belt components for overload-related stress."
        )


    elif scenario == "Misalignment":

        recommendation = (
            "Inspect belt tracking, idlers, pulleys and conveyor "
            "alignment for abnormal mechanical conditions."
        )


    else:

        recommendation = (
            "Perform preventive inspection."
        )


    st.info(
        "🔧 " + recommendation
    )


    # =====================================================
    # FINAL DIAGNOSIS
    # =====================================================

    st.divider()

    st.header("📝 Final Diagnosis")


    if condition == "NORMAL":

        st.success(
            "🟢 NORMAL: Conveyor is operating within "
            "the defined operating limits."
        )


    elif condition == "WARNING":

        st.warning(
            "🟡 WARNING: Abnormal operating behavior detected. "
            "Preventive inspection is recommended."
        )


    else:

        st.error(
            "🔴 CRITICAL: Significant abnormality detected. "
            "Immediate conveyor inspection is recommended."
        )


    # =====================================================
    # METHODOLOGY
    # =====================================================

    st.divider()

    st.header("⚙️ Simulation Methodology")


    st.write(
        """
        **Sensor Inputs**
        
        Vibration + RPM + Motor Current + Load
        
        ↓
        
        **Signal Simulation**
        
        Simulated operating/fault conditions
        
        ↓
        
        **Vibration Signal Processing**
        
        RMS + FFT
        
        ↓
        
        **Rule-Based Multi-Sensor Fusion**
        
        Predefined thresholds + weighted risk contribution
        
        ↓
        
        **Health & Failure-Risk Assessment**
        
        Health Score + Risk %
        
        ↓
        
        **Fault Detection**
        
        Belt joint deterioration / belt damage / belt slip /
        overload / misalignment
        
        ↓
        
        **Maintenance Recommendation**
        
        Suggested inspection/action
        """
    )


