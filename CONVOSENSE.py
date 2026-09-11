import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ConvoSense | Intelligent Conveyor Health",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ ConvoSense — Intelligent Conveyor Belt Health Monitoring")
st.caption(
    "Simulation-based conveyor monitoring, visual belt tracking, "
    "multi-sensor analysis and predictive maintenance assessment"
)


# =========================================================
# CONSTANTS
# =========================================================

NORMAL_VIBRATION = 2.0       # mm/s
NORMAL_RPM = 1000.0          # RPM
NORMAL_CURRENT = 8.0         # A
NORMAL_LOAD = 100.0          # kg

VIBRATION_WARNING = 4.0
CURRENT_WARNING = 12.0
LOAD_WARNING = 80.0

CAMERA_WARNING = 10.0        # %
CAMERA_CRITICAL = 20.0       # %

CAMERA_INTERVAL = 3

# Sensor health weights
W_VIBRATION = 0.35
W_RPM = 0.20
W_CURRENT = 0.25
W_LOAD = 0.20

# Multimodal fusion weights
SENSOR_FUSION_WEIGHT = 0.80
CAMERA_FUSION_WEIGHT = 0.20


# =========================================================
# SESSION STATE
# =========================================================

if "camera_count" not in st.session_state:
    st.session_state.camera_count = 0

if "camera_history" not in st.session_state:
    st.session_state.camera_history = []

if "motion_offset" not in st.session_state:
    st.session_state.motion_offset = 0

if "sensor_history" not in st.session_state:
    st.session_state.sensor_history = []


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clamp(value, low=0, high=100):
    return max(low, min(high, value))


def vibration_health(v):
    """
    Reference:
    2 mm/s = healthy baseline.
    Increasing vibration progressively reduces health.
    """
    health = 100 - ((v / NORMAL_VIBRATION) - 1) * 50
    return clamp(health)


def rpm_health(rpm):
    deviation = abs(rpm - NORMAL_RPM) / NORMAL_RPM * 100
    return clamp(100 - deviation)


def current_health(current):
    health = 100 - ((current / NORMAL_CURRENT) - 1) * 50
    return clamp(health)


def load_health(load):
    utilization = load / NORMAL_LOAD * 100
    health = 100 - utilization * 0.50
    return clamp(health)


def vibration_status(v):
    if v <= VIBRATION_WARNING:
        return "NORMAL"
    elif v <= 7:
        return "WARNING"
    else:
        return "CRITICAL"


def rpm_status(rpm):
    deviation = abs(rpm - NORMAL_RPM) / NORMAL_RPM * 100

    if deviation <= 5:
        return "NORMAL"
    elif deviation <= 15:
        return "WARNING"
    else:
        return "CRITICAL"


def current_status(current):
    if current <= CURRENT_WARNING:
        return "NORMAL"
    elif current <= 16:
        return "WARNING"
    else:
        return "CRITICAL"


def load_status(load):
    if load <= LOAD_WARNING:
        return "NORMAL"
    elif load <= 95:
        return "WARNING"
    else:
        return "CRITICAL"


def status_color(status):
    if status == "NORMAL":
        return "green"
    elif status == "WARNING":
        return "orange"
    return "red"


# =========================================================
# CAMERA SIMULATION
# =========================================================

def get_camera_deviation():
    """
    Simulated vertical belt-tracking measurement.

    IMPORTANT:
    The CENTERLINE is the fixed reference.

    Positive deviation  = belt moves UPWARD from centerline
    Negative deviation  = belt moves DOWNWARD from centerline
    Zero deviation      = belt remains CENTERED

    The absolute deviation determines the severity:

        0% to 10%       -> NORMAL
        >10% to 20%     -> WARNING
        >20%            -> CRITICAL

    Both upward and downward movement are treated as
    misalignment when the deviation exceeds the warning threshold.
    """

    # -----------------------------------------------------
    # TEST CASE SEQUENCE
    # -----------------------------------------------------
    #
    # 0      = Centered / Normal
    # +5     = Small upward movement / Normal
    # -5     = Small downward movement / Normal
    # +10    = Boundary / Normal
    # -10    = Boundary / Normal
    # +12    = Upward misalignment / WARNING
    # -12    = Downward misalignment / WARNING
    # +15    = Upward misalignment / WARNING
    # -15    = Downward misalignment / WARNING
    # +20    = Boundary / WARNING
    # -20    = Boundary / WARNING
    # +23    = Severe upward misalignment / CRITICAL
    # -23    = Severe downward misalignment / CRITICAL
    # +30    = Severe upward misalignment / CRITICAL
    # -30    = Severe downward misalignment / CRITICAL
    #
    # This allows the dashboard to visibly demonstrate
    # every possible camera condition.

    sequence = [
        0,
        5,
        -5,
        10,
        -10,
        12,
        -12,
        15,
        -15,
        20,
        -20,
        23,
        -23,
        30,
        -30,
        4,
        -4,
        0
    ]

    index = st.session_state.camera_count % len(sequence)

    return sequence[index]


def camera_classification(deviation):

    # -----------------------------------------------------
    # ABSOLUTE DEVIATION
    # -----------------------------------------------------
    #
    # The magnitude determines whether the belt is
    # normal, warning or critical.
    #
    # Direction is handled separately below.

    abs_dev = abs(deviation)

    if abs_dev <= CAMERA_WARNING:
        classification = "NORMAL"

    elif abs_dev <= CAMERA_CRITICAL:
        classification = "WARNING"

    else:
        classification = "CRITICAL"

    # -----------------------------------------------------
    # VERTICAL MISALIGNMENT DIRECTION
    # -----------------------------------------------------
    #
    # Centerline = reference
    #
    # Positive = upward
    # Negative = downward
    #
    # Small movement around centerline is considered centered.

    if deviation > 3:
        direction = "UPWARD"

    elif deviation < -3:
        direction = "DOWNWARD"

    else:
        direction = "CENTERED"

    # -----------------------------------------------------
    # VISUAL HEALTH
    # -----------------------------------------------------

    if classification == "NORMAL":
        visual_health = 100 - abs_dev * 1.5

    else:
        visual_health = 100 - abs_dev * 2.0

    visual_health = clamp(visual_health)

    return classification, direction, visual_health


# =========================================================
# MOVING CONVEYOR IMAGE
# =========================================================

def create_moving_conveyor_frame(deviation, capture_number):

    width = 1100
    height = 620

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    # -----------------------------------------------------
    # BACKGROUND
    # -----------------------------------------------------

    ax.set_facecolor("#dfe3e6")

    # Industrial floor
    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            width,
            height,
            facecolor="#dfe3e6"
        )
    )

    # -----------------------------------------------------
    # STRUCTURAL FRAME
    # -----------------------------------------------------

    frame_y = 100
    frame_height = 420

    ax.add_patch(
        plt.Rectangle(
            (120, frame_y),
            860,
            frame_height,
            fill=False,
            linewidth=16
        )
    )

    # Cross supports
    for x in [220, 420, 620, 820]:

        ax.plot(
            [x, x],
            [100, 520],
            linewidth=8
        )

    # -----------------------------------------------------
    # CONVEYOR BELT
    # -----------------------------------------------------

    belt_width = 650
    belt_height = 300

    base_center_x = width / 2

    # -----------------------------------------------------
    # REFERENCE VERTICAL POSITION
    # -----------------------------------------------------

    # THIS IS THE FIXED CENTERLINE REFERENCE.

    base_center_y = 310

    # -----------------------------------------------------
    # VERTICAL BELT SHIFT
    # -----------------------------------------------------

    # Positive deviation = upward
    # Negative deviation = downward

    belt_shift_y = deviation * 6

    belt_center_y = base_center_y + belt_shift_y

    belt_left = base_center_x - belt_width / 2

    belt_bottom = belt_center_y - belt_height / 2

    # -----------------------------------------------------
    # BELT BODY
    # -----------------------------------------------------

    ax.add_patch(
        plt.Rectangle(
            (belt_left, belt_bottom),
            belt_width,
            belt_height,
            facecolor="#151515",
            edgecolor="black",
            linewidth=5
        )
    )

    # -----------------------------------------------------
    # YELLOW TRACKING STRIPES
    # -----------------------------------------------------

    stripe_width = 14

    ax.add_patch(
        plt.Rectangle(
            (
                belt_left + 20,
                belt_bottom + 8
            ),
            stripe_width,
            belt_height - 16,
            facecolor="#f2c500"
        )
    )

    ax.add_patch(
        plt.Rectangle(
            (
                belt_left + belt_width - 34,
                belt_bottom + 8
            ),
            stripe_width,
            belt_height - 16,
            facecolor="#f2c500"
        )
    )

    # -----------------------------------------------------
    # MOVING BELT TREAD
    # -----------------------------------------------------

    motion = st.session_state.motion_offset

    for x in np.arange(
        belt_left + 60,
        belt_left + belt_width - 30,
        45
    ):

        shifted_x = x + (motion % 45)

        if shifted_x > belt_left + belt_width - 30:
            shifted_x -= belt_width - 90

        ax.plot(
            [shifted_x, shifted_x],
            [belt_bottom + 25, belt_bottom + belt_height - 25],
            linewidth=2,
            alpha=0.35
        )

    # -----------------------------------------------------
    # IRON ORE / MATERIAL MOVEMENT
    # -----------------------------------------------------

    rng = np.random.default_rng(100 + capture_number)

    for i in range(35):

        x = rng.uniform(
            belt_left + 50,
            belt_left + belt_width - 50
        )

        base_y = rng.uniform(
            belt_bottom + 30,
            belt_bottom + belt_height - 30
        )

        # Material continues moving FORWARD horizontally.
        # This movement is normal and is not treated as
        # vertical belt misalignment.

        moving_x = (
            x +
            (motion * 1.8) % (belt_width - 100)
        )

        if moving_x > belt_left + belt_width - 50:
            moving_x -= belt_width - 100

        size = rng.uniform(5, 13)

        ax.scatter(
            moving_x,
            base_y,
            s=size * 7,
            alpha=0.8
        )

    # -----------------------------------------------------
    # REFERENCE HORIZONTAL ALIGNMENT LINE
    # -----------------------------------------------------

    # FIXED CENTERLINE.
    #
    # This line NEVER moves.
    #
    # The belt centerline moves relative to this line.
    #
    # Therefore:
    #
    # Reference line = expected belt center
    # Detected line  = actual belt center
    #
    # Distance between them = vertical deviation.

    reference_y = base_center_y

    ax.plot(
        [belt_left - 100, belt_left + belt_width + 100],
        [reference_y, reference_y],
        linestyle="--",
        linewidth=3,
        label="Reference Centerline"
    )

    # -----------------------------------------------------
    # DETECTED BELT CENTERLINE
    # -----------------------------------------------------

    detected_y = belt_center_y

    ax.plot(
        [belt_left - 100, belt_left + belt_width + 100],
        [detected_y, detected_y],
        linestyle="-",
        linewidth=4,
        label="Detected Belt Centerline"
    )

    # -----------------------------------------------------
    # VERTICAL DEVIATION ARROW
    # -----------------------------------------------------

    ax.annotate(
        "",
        xy=(base_center_x + 260, detected_y),
        xytext=(base_center_x + 260, reference_y),
        arrowprops=dict(
            arrowstyle="<->",
            linewidth=3
        )
    )

    # -----------------------------------------------------
    # DEVIATION LABEL
    # -----------------------------------------------------

    label_y = (
        reference_y + detected_y
    ) / 2

    ax.text(
        base_center_x + 290,
        label_y,
        f"Vertical Deviation = {abs(deviation):.1f}%",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold"
    )

    # -----------------------------------------------------
    # CAMERA INSPECTION ZONE
    # -----------------------------------------------------

    zone_width = 420
    zone_height = 230

    # The inspection zone remains referenced to the original
    # conveyor alignment so vertical movement can be observed.

    zone_left = base_center_x - zone_width / 2
    zone_bottom = base_center_y - zone_height / 2

    ax.add_patch(
        plt.Rectangle(
            (
                zone_left,
                zone_bottom
            ),
            zone_width,
            zone_height,
            fill=False,
            linestyle="--",
            linewidth=3
        )
    )

    ax.text(
        base_center_x,
        zone_bottom + zone_height + 10,
        "CAMERA INSPECTION ZONE",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

    # -----------------------------------------------------
    # OVERHEAD CAMERA
    # -----------------------------------------------------

    camera_x = base_center_x
    camera_y = 575

    ax.add_patch(
        plt.Rectangle(
            (
                camera_x - 55,
                camera_y - 30
            ),
            110,
            45,
            facecolor="#30343b",
            edgecolor="black",
            linewidth=2
        )
    )

    # Camera lens
    ax.scatter(
        camera_x,
        camera_y - 8,
        s=130,
        marker="o"
    )

    # Blue indicator
    ax.scatter(
        camera_x + 40,
        camera_y - 8,
        s=50,
        marker="o"
    )

    # Light cone
    ax.fill(
        [
            camera_x - 30,
            camera_x + 30,
            camera_x + 190,
            camera_x - 190
        ],
        [
            camera_y - 35,
            camera_y - 35,
            360,
            360
        ],
        alpha=0.08
    )

    # -----------------------------------------------------
    # LABELS
    # -----------------------------------------------------

    ax.text(
        30,
        590,
        "CONVOSENSE — LIVE CONVEYOR VISUAL MONITORING",
        fontsize=16,
        fontweight="bold"
    )

    ax.text(
        30,
        40,
        "Simulated overhead camera inspection",
        fontsize=10
    )

    ax.text(
        width - 30,
        40,
        f"CAPTURE #{capture_number}",
        fontsize=11,
        ha="right",
        fontweight="bold"
    )

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2
    )

    plt.tight_layout()

    return fig


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("📡 Live Sensor Controls")


# =========================================================
# SENSOR INPUTS
# =========================================================

st.sidebar.subheader("📡 Live Sensor Inputs")

manual_vibration = st.sidebar.slider(
    "Vibration (mm/s)",
    0.0,
    12.0,
    NORMAL_VIBRATION,
    0.1
)

manual_rpm = st.sidebar.slider(
    "RPM",
    500.0,
    1200.0,
    NORMAL_RPM,
    10.0
)

manual_current = st.sidebar.slider(
    "Motor Current (A)",
    0.0,
    20.0,
    NORMAL_CURRENT,
    0.1
)

manual_load = st.sidebar.slider(
    "Load (kg)",
    0.0,
    NORMAL_LOAD,
    50.0,
    1.0
)


if st.sidebar.button("🔄 Clear History"):

    st.session_state.camera_history = []
    st.session_state.sensor_history = []
    st.session_state.camera_count = 0
    st.session_state.motion_offset = 0


# =========================================================
# LIVE SENSOR VALUES
# =========================================================

vibration = manual_vibration
rpm = manual_rpm
current = manual_current
load = manual_load


# =========================================================
# TOP INFORMATION
# =========================================================

st.info(
    f"**Live Sensor Inputs:**  "
    f"Vibration: {vibration:.1f} mm/s  |  "
    f"RPM: {rpm:.0f}  |  "
    f"Current: {current:.1f} A  |  "
    f"Load: {load:.0f} kg"
)


# =========================================================
# 1. LIVE MOVING CONVEYOR
# =========================================================

st.header("1️⃣ 🎥 LIVE MOVING CONVEYOR")

st.write(
    "The conveyor movement, belt tracking and material flow are "
    "simulated for prototype demonstration."
)


# =========================================================
# 2–6 CAMERA MODULE
# =========================================================

st.header("2️⃣ 📷 CAMERA-BASED VISUAL DETECTION")


@st.fragment(run_every=CAMERA_INTERVAL)
def live_camera_module():

    # Capture counter
    st.session_state.camera_count += 1

    capture_number = st.session_state.camera_count

    # Motion update
    st.session_state.motion_offset += 55

    # Get simulated vertical camera deviation
    deviation = get_camera_deviation()

    classification, direction, visual_health = camera_classification(
        deviation
    )

    # Timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Create frame
    fig = create_moving_conveyor_frame(
        deviation,
        capture_number
    )

    # Store history
    history_entry = {
        "Capture": capture_number,
        "Time": timestamp,
        "Vertical Deviation (%)": round(abs(deviation), 1),
        "Direction": direction,
        "Visual Health (%)": round(visual_health, 1),
        "Result": classification
    }

    st.session_state.camera_history.append(
        history_entry
    )

    # Keep last 20 captures
    st.session_state.camera_history = (
        st.session_state.camera_history[-20:]
    )

    # -----------------------------------------------------
    # CAMERA IMAGE
    # -----------------------------------------------------

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.caption(
        "Simulated overhead camera inspection — the fixed centerline "
        "is used as the alignment reference. Forward belt movement is "
        "treated as normal. Vertical displacement in either direction "
        "is monitored as potential misalignment. Automatic frame "
        "capture occurs every 3 seconds."
    )

    # -----------------------------------------------------
    # CAMERA METRICS
    # -----------------------------------------------------

    st.subheader("3️⃣ 📐 VERTICAL BELT ALIGNMENT / CENTERLINE TRACING")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Detected Vertical Deviation",
        f"{abs(deviation):.1f}%"
    )

    c2.metric(
        "Direction",
        direction
    )

    c3.metric(
        "Visual Health",
        f"{visual_health:.1f}%"
    )

    c4.metric(
        "Capture Number",
        f"#{capture_number}"
    )

    # -----------------------------------------------------
    # ALIGNMENT STATUS
    # -----------------------------------------------------

    st.subheader("📏 BELT ALIGNMENT STATUS")

    if abs(deviation) <= CAMERA_WARNING:

        st.success(
            f"🟢 CENTERED / NORMAL — Belt centerline is within the "
            f"acceptable alignment range. Deviation = {abs(deviation):.1f}%"
        )

    elif abs(deviation) <= CAMERA_CRITICAL:

        st.warning(
            f"🟠 MISALIGNMENT WARNING — Belt has shifted "
            f"{direction.lower()} from the reference centerline. "
            f"Deviation = {abs(deviation):.1f}%"
        )

    else:

        st.error(
            f"🔴 CRITICAL MISALIGNMENT — Severe belt displacement detected "
            f"{direction.lower()} from the reference centerline. "
            f"Deviation = {abs(deviation):.1f}%"
        )

    # -----------------------------------------------------
    # CAMERA RESULT
    # -----------------------------------------------------

    st.subheader("6️⃣ 🔍 CAMERA DETECTION RESULT")

    if classification == "NORMAL":

        st.success(
            f"🟢 NORMAL — Belt remains within acceptable vertical "
            f"alignment. Vertical deviation = {abs(deviation):.1f}%"
        )

    elif classification == "WARNING":

        st.warning(
            f"🟠 WARNING — Vertical belt displacement detected. "
            f"Direction = {direction}, "
            f"Vertical deviation = {abs(deviation):.1f}%"
        )

    else:

        st.error(
            f"🔴 CRITICAL — Severe vertical belt misalignment detected. "
            f"Direction = {direction}, "
            f"Vertical deviation = {abs(deviation):.1f}%"
        )

    # -----------------------------------------------------
    # CAPTURE HISTORY
    # -----------------------------------------------------

    st.subheader("5️⃣ 📸 CAMERA CAPTURE HISTORY")

    history_df = pd.DataFrame(
        st.session_state.camera_history
    )

    if not history_df.empty:

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )


live_camera_module()


# =========================================================
# CURRENT CAMERA STATE FOR OTHER MODULES
# =========================================================

if st.session_state.camera_history:

    latest_camera = (
        st.session_state.camera_history[-1]
    )

    camera_health = latest_camera[
        "Visual Health (%)"
    ]

    camera_result = latest_camera[
        "Result"
    ]

    camera_direction = latest_camera[
        "Direction"
    ]

    camera_deviation = latest_camera[
        "Vertical Deviation (%)"
    ]

else:

    camera_health = 100
    camera_result = "NORMAL"
    camera_direction = "CENTERED"
    camera_deviation = 0


# =========================================================
# 7. MONITORING RESULTS
# =========================================================

st.header("7️⃣ 📊 MONITORING RESULTS")

st.caption(
    "Every metric below is recalculated directly from the current live sensor inputs."
)

load_utilization = (
    load / NORMAL_LOAD
) * 100

v_health = vibration_health(vibration)
r_health = rpm_health(rpm)
c_health = current_health(current)
l_health = load_health(load)

sensor_health = (
    W_VIBRATION * v_health +
    W_RPM * r_health +
    W_CURRENT * c_health +
    W_LOAD * l_health
)

overall_health = (
    SENSOR_FUSION_WEIGHT * sensor_health +
    CAMERA_FUSION_WEIGHT * camera_health
)

failure_risk = clamp(
    100 - overall_health
)


# Save sensor history
st.session_state.sensor_history.append(
    {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Vibration": vibration,
        "RPM": rpm,
        "Current": current,
        "Load": load,
        "Health": overall_health
    }
)

st.session_state.sensor_history = (
    st.session_state.sensor_history[-50:]
)


# =========================================================
# MONITORING METRICS
# =========================================================

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Vibration",
    f"{vibration:.1f} mm/s"
)

m2.metric(
    "RPM",
    f"{rpm:.0f}"
)

m3.metric(
    "Motor Current",
    f"{current:.1f} A"
)

m4.metric(
    "Load",
    f"{load:.0f} kg"
)

st.info(
    f"**Load Utilization = {load:.1f} / {NORMAL_LOAD:.0f} × 100 = "
    f"{load_utilization:.1f}%**"
)


# =========================================================
# 8. LIVE CONDITION INTERPRETATION
# =========================================================

st.header("8️⃣ 🧭 LIVE CONDITION INTERPRETATION")

live_flags = []

if vibration > VIBRATION_WARNING:
    live_flags.append("Elevated vibration")

if abs(rpm - NORMAL_RPM) > 100:
    live_flags.append("Significant RPM deviation")

if current > CURRENT_WARNING:
    live_flags.append("High motor current")

if load > LOAD_WARNING:
    live_flags.append("High conveyor load")

if camera_result in ["WARNING", "CRITICAL"]:
    live_flags.append("Vertical belt tracking deviation")


if not live_flags:

    st.success(
        "🟢 LIVE CONDITION: NORMAL — Current sensor values and camera tracking "
        "are within the simulated acceptable range."
    )

else:

    st.warning(
        "⚠️ LIVE CONDITION: ABNORMAL — " +
        ", ".join(live_flags) +
        ". The downstream health, risk, fault and maintenance results "
        "have been recalculated from the current inputs."
    )


# =========================================================
# 9. VIBRATION SIGNAL ANALYSIS
# =========================================================

st.header("9️⃣ 📈 VIBRATION SIGNAL ANALYSIS")

Fs = 1000
duration = 2

t = np.arange(
    0,
    duration,
    1 / Fs
)

rng = np.random.default_rng(
    42 + st.session_state.camera_count
)

noise = rng.normal(
    0,
    0.25,
    len(t)
)


rpm_frequency = max(
    5.0,
    rpm / 60.0
)

signal = (
    vibration
    + (0.18 * vibration) *
    np.sin(
        2 * np.pi *
        rpm_frequency *
        t
    )
    + (0.08 * vibration) *
    np.sin(
        2 * np.pi *
        2 *
        rpm_frequency *
        t
    )
    + noise
)


# RMS
vibration_rms = np.sqrt(
    np.mean(signal ** 2)
)

st.metric(
    "Calculated Vibration RMS",
    f"{vibration_rms:.2f} mm/s"
)

fig_vib, ax_vib = plt.subplots(
    figsize=(12, 4)
)

ax_vib.plot(
    t[:3000],
    signal[:3000]
)

ax_vib.set_title(
    "Simulated Vibration Time-Domain Signal"
)

ax_vib.set_xlabel(
    "Time (s)"
)

ax_vib.set_ylabel(
    "Vibration (mm/s)"
)

ax_vib.grid(True)

st.pyplot(
    fig_vib,
    use_container_width=True
)

plt.close(fig_vib)


# =========================================================
# 10. FFT FREQUENCY ANALYSIS
# =========================================================

st.header("🔬 10️⃣ FFT FREQUENCY ANALYSIS")

N = len(signal)

fft_values = np.fft.rfft(
    signal - np.mean(signal)
)

frequencies = np.fft.rfftfreq(
    N,
    1 / Fs
)

magnitude = (
    2 / N
) * np.abs(fft_values)

dominant_index = np.argmax(
    magnitude[1:]
) + 1

dominant_frequency = frequencies[
    dominant_index
]

dominant_amplitude = magnitude[
    dominant_index
]

f1, f2 = st.columns(2)

f1.metric(
    "Dominant Frequency",
    f"{dominant_frequency:.2f} Hz"
)

f2.metric(
    "Dominant Amplitude",
    f"{dominant_amplitude:.3f}"
)

fig_fft, ax_fft = plt.subplots(
    figsize=(12, 4)
)

ax_fft.plot(
    frequencies,
    magnitude
)

ax_fft.set_xlim(
    0,
    300
)

ax_fft.set_title(
    "FFT Frequency Spectrum"
)

ax_fft.set_xlabel(
    "Frequency (Hz)"
)

ax_fft.set_ylabel(
    "Magnitude"
)

ax_fft.grid(True)

st.pyplot(
    fig_fft,
    use_container_width=True
)

plt.close(fig_fft)

st.caption(
    "FFT converts the vibration signal from the time domain into "
    "the frequency domain to identify dominant periodic components."
)


# =========================================================
# 11. MULTI-SENSOR ANALYSIS
# =========================================================

st.header("1️⃣1️⃣ 🔍 MULTI-SENSOR ANALYSIS")

sensor_df = pd.DataFrame(
    {
        "Parameter": [
            "Vibration",
            "RPM",
            "Motor Current",
            "Load"
        ],

        "Measured Value": [
            vibration,
            rpm,
            current,
            load
        ],

        "Normal Reference": [
            NORMAL_VIBRATION,
            NORMAL_RPM,
            NORMAL_CURRENT,
            NORMAL_LOAD
        ],

        "Health (%)": [
            v_health,
            r_health,
            c_health,
            l_health
        ]
    }
)

st.dataframe(
    sensor_df.style.format(
        {
            "Measured Value": "{:.2f}",
            "Normal Reference": "{:.2f}",
            "Health (%)": "{:.1f}"
        }
    ),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 12. SENSOR TRENDS
# =========================================================

st.header("1️⃣2️⃣ 📊 SENSOR TRENDS")

trend_df = pd.DataFrame(
    st.session_state.sensor_history
)

if not trend_df.empty:

    # Vibration
    fig1, ax1 = plt.subplots(
        figsize=(12, 3)
    )

    ax1.plot(
        trend_df.index,
        trend_df["Vibration"],
        marker="o"
    )

    ax1.axhline(
        VIBRATION_WARNING,
        linestyle="--",
        label="Warning Threshold"
    )

    ax1.set_title(
        "Vibration Trend"
    )

    ax1.set_ylabel(
        "mm/s"
    )

    ax1.legend()

    ax1.grid(True)

    st.pyplot(
        fig1,
        use_container_width=True
    )

    plt.close(fig1)

    # RPM
    fig2, ax2 = plt.subplots(
        figsize=(12, 3)
    )

    ax2.plot(
        trend_df.index,
        trend_df["RPM"],
        marker="o"
    )

    ax2.axhline(
        NORMAL_RPM,
        linestyle="--",
        label="Normal RPM"
    )

    ax2.set_title(
        "RPM Trend"
    )

    ax2.set_ylabel(
        "RPM"
    )

    ax2.legend()

    ax2.grid(True)

    st.pyplot(
        fig2,
        use_container_width=True
    )

    plt.close(fig2)

    # Current
    fig3, ax3 = plt.subplots(
        figsize=(12, 3)
    )

    ax3.plot(
        trend_df.index,
        trend_df["Current"],
        marker="o"
    )

    ax3.axhline(
        CURRENT_WARNING,
        linestyle="--",
        label="Warning Threshold"
    )

    ax3.set_title(
        "Motor Current Trend"
    )

    ax3.set_ylabel(
        "A"
    )

    ax3.legend()

    ax3.grid(True)

    st.pyplot(
        fig3,
        use_container_width=True
    )

    plt.close(fig3)

    # Load
    fig4, ax4 = plt.subplots(
        figsize=(12, 3)
    )

    ax4.plot(
        trend_df.index,
        trend_df["Load"],
        marker="o"
    )

    ax4.axhline(
        LOAD_WARNING,
        linestyle="--",
        label="Warning Threshold"
    )

    ax4.set_title(
        "Load Trend"
    )

    ax4.set_ylabel(
        "kg"
    )

    ax4.legend()

    ax4.grid(True)

    st.pyplot(
        fig4,
        use_container_width=True
    )

    plt.close(fig4)


# =========================================================
# 13. MULTI-MODAL HEALTH ASSESSMENT
# =========================================================

st.header("1️⃣3️⃣ 🧠 MULTI-MODAL HEALTH ASSESSMENT")

st.write(
    "The final conveyor health score combines the four physical "
    "sensor channels with the simulated camera inspection result."
)

h1, h2, h3 = st.columns(3)

h1.metric(
    "Camera Health",
    f"{camera_health:.1f}%"
)

h2.metric(
    "Sensor Health",
    f"{sensor_health:.1f}%"
)

h3.metric(
    "Overall Conveyor Health",
    f"{overall_health:.1f}%"
)

st.progress(
    int(overall_health)
)


# Fusion contribution

col1, col2 = st.columns(2)

with col1:

    st.info(
        f"""
        **Sensor Contribution**

        Sensor Health = **{sensor_health:.1f}%**

        Weight = **80%**

        Contribution =
        {sensor_health:.1f} × 0.80 =
        **{sensor_health * 0.80:.1f}**
        """
    )

with col2:

    st.info(
        f"""
        **Camera Contribution**

        Camera Health = **{camera_health:.1f}%**

        Weight = **20%**

        Contribution =
        {camera_health:.1f} × 0.20 =
        **{camera_health * 0.20:.1f}**
        """
    )

st.success(
    f"🧠 **FUSED OVERALL CONVEYOR HEALTH = {overall_health:.1f}%**"
)


# =========================================================
# 14. FAILURE RISK
# =========================================================

st.header("1️⃣4️⃣ 🚨 FAILURE-RISK ASSESSMENT")

risk_index = failure_risk

if risk_index <= 20:

    risk_level = "LOW"

elif risk_index <= 45:

    risk_level = "MODERATE"

elif risk_index <= 70:

    risk_level = "HIGH"

else:

    risk_level = "CRITICAL"


r1, r2, r3 = st.columns(3)

r1.metric(
    "Failure-Risk Index",
    f"{risk_index:.1f}%"
)

r2.metric(
    "Risk Level",
    risk_level
)

r3.metric(
    "Overall Health",
    f"{overall_health:.1f}%"
)

st.progress(
    int(risk_index)
)


if risk_level == "LOW":

    st.success(
        f"🟢 LOW RISK — Current operating condition is within the simulated healthy range. "
        f"Failure-Risk Index = {risk_index:.1f}%."
    )

elif risk_level == "MODERATE":

    st.warning(
        f"🟠 MODERATE RISK — Abnormal indicators are emerging. "
        f"Failure-Risk Index = {risk_index:.1f}%."
    )

elif risk_level == "HIGH":

    st.warning(
        f"🟠 HIGH RISK — Significant abnormality detected. "
        f"Inspection and preventive maintenance are recommended. "
        f"Failure-Risk Index = {risk_index:.1f}%."
    )

else:

    st.error(
        f"🔴 CRITICAL RISK — Multiple abnormal indicators require immediate inspection. "
        f"Failure-Risk Index = {risk_index:.1f}%."
    )


with st.expander(
    "🔬 How is the Failure-Risk Index calculated?"
):

    st.write(
        f"""
        The prototype defines a simulated Failure-Risk Index as the
        complement of the fused conveyor health:

        **Failure-Risk Index = 100 − Overall Conveyor Health**

        Current values:

        • Overall Conveyor Health = **{overall_health:.1f}%**

        • Failure-Risk Index = **{risk_index:.1f}%**

        This is a **simulation-based risk indicator**, not a statistically
        calibrated probability of failure. A deployed system would require
        historical conveyor failure data for probability calibration.
        """
    )


# =========================================================
# 15. BELT / JOINT DAMAGE SEVERITY INDEX
# =========================================================

st.header("1️⃣5️⃣ 🩺 BELT / JOINT DAMAGE SEVERITY INDEX")

# Normalize sensor abnormality

vibration_severity = clamp(
    (
        (vibration - NORMAL_VIBRATION) /
        (10 - NORMAL_VIBRATION)
    ) * 100
)

rpm_severity = clamp(
    abs(rpm - NORMAL_RPM) /
    NORMAL_RPM * 100
)

current_severity = clamp(
    (
        (current - NORMAL_CURRENT) /
        (20 - NORMAL_CURRENT)
    ) * 100
)

load_severity = clamp(
    (
        (load - NORMAL_LOAD) /
        NORMAL_LOAD
    ) * 100
)

camera_severity = clamp(
    camera_deviation /
    CAMERA_CRITICAL * 100
)

damage_severity = (
    0.30 * vibration_severity +
    0.15 * rpm_severity +
    0.20 * current_severity +
    0.15 * load_severity +
    0.20 * camera_severity
)

damage_severity = clamp(
    damage_severity
)


if damage_severity <= 20:

    damage_level = "LOW"

elif damage_severity <= 45:

    damage_level = "MODERATE"

elif damage_severity <= 70:

    damage_level = "HIGH"

else:

    damage_level = "CRITICAL"


d1, d2 = st.columns(2)

d1.metric(
    "Damage Severity Index",
    f"{damage_severity:.1f}%"
)

d2.metric(
    "Severity Level",
    damage_level
)

st.progress(
    int(damage_severity)
)

st.caption(
    "The Damage Severity Index is a weighted simulation indicator "
    "combining abnormal sensor conditions and camera deviation."
)


# =========================================================
# 16. FAULT DETECTION
# =========================================================

st.header("1️⃣6️⃣ 🚨 FAULT DETECTION")

faults = []

if vibration > VIBRATION_WARNING:

    faults.append(
        "Elevated vibration — possible idler/bearing/mechanical abnormality"
    )


if abs(rpm - NORMAL_RPM) > 100:

    faults.append(
        "RPM deviation — possible belt slip, drive or speed abnormality"
    )


if current > CURRENT_WARNING:

    faults.append(
        "High motor current — possible overload or increased mechanical resistance"
    )


if load > LOAD_WARNING:

    faults.append(
        "High conveyor load — increased loading condition"
    )


if camera_result == "WARNING":

    faults.append(
        f"Visual vertical belt misalignment warning — belt shifted {camera_direction}"
    )


if camera_result == "CRITICAL":

    faults.append(
        f"Critical visual vertical belt misalignment — severe shift {camera_direction}"
    )


if not faults:

    st.success(
        "🟢 No significant abnormality detected by the current simulation."
    )

else:

    for fault in faults:

        st.warning(
            f"⚠️ {fault}"
        )


# =========================================================
# 17. MAINTENANCE RECOMMENDATION
# =========================================================

st.header("1️⃣7️⃣ 🛠️ MAINTENANCE RECOMMENDATION")

recommendations = []


if vibration > VIBRATION_WARNING:

    recommendations.append(
        "Inspect idler rollers, bearings, mounting points and mechanical looseness."
    )


if abs(rpm - NORMAL_RPM) > 100:

    recommendations.append(
        "Inspect belt traction, drive pulley, motor coupling and speed transmission."
    )


if current > CURRENT_WARNING:

    recommendations.append(
        "Check motor loading, drive system, material accumulation and mechanical resistance."
    )


if load > LOAD_WARNING:

    recommendations.append(
        "Check material loading and ensure conveyor operation remains within allowable load."
    )


if camera_result in ["WARNING", "CRITICAL"]:

    recommendations.append(
        "Inspect belt tracking, idler alignment, pulley alignment and belt edges."
    )


if not recommendations:

    recommendations.append(
        "Continue routine condition monitoring and preventive maintenance."
    )


for i, recommendation in enumerate(
    recommendations,
    start=1
):

    st.info(
        f"**{i}.** {recommendation}"
    )


# =========================================================
# FINAL DIAGNOSIS
# =========================================================

st.header("1️⃣8️⃣ 📝 FINAL DIAGNOSIS")


# Determine final condition

if overall_health >= 80:

    diagnosis_status = "NORMAL"

    diagnosis_message = (
        "Conveyor operating condition is currently stable. "
        "Continue condition monitoring."
    )

    box_bg = "#eaf7ea"
    box_border = "#2e8b57"

elif overall_health >= 60:

    diagnosis_status = "WARNING"

    diagnosis_message = (
        "Abnormal indicators are present. Preventive inspection "
        "is recommended before the condition escalates."
    )

    box_bg = "#fff8e1"
    box_border = "#d99a00"

else:

    diagnosis_status = "CRITICAL"

    diagnosis_message = (
        "Multiple abnormal indicators suggest elevated conveyor risk. "
        "Immediate inspection and maintenance action are recommended."
    )

    box_bg = "#fdecec"
    box_border = "#d32f2f"


# =========================================================
# FINAL DIAGNOSIS DISPLAY
# =========================================================

if diagnosis_status == "NORMAL":

    st.success(
        f"🟢 FINAL DIAGNOSIS: {diagnosis_status}\n\n"
        f"{diagnosis_message}"
    )

elif diagnosis_status == "WARNING":

    st.warning(
        f"🟠 FINAL DIAGNOSIS: {diagnosis_status}\n\n"
        f"{diagnosis_message}"
    )

else:

    st.error(
        f"🔴 FINAL DIAGNOSIS: {diagnosis_status}\n\n"
        f"{diagnosis_message}"
    )


# =========================================================
# DIAGNOSIS SUMMARY
# =========================================================

d1, d2, d3, d4 = st.columns(4)

d1.metric(
    "Overall Health",
    f"{overall_health:.1f}%"
)

d2.metric(
    "Failure Risk",
    f"{risk_index:.1f}%"
)

d3.metric(
    "Camera Result",
    camera_result
)

d4.metric(
    "Damage Severity",
    f"{damage_severity:.1f}%"
)


# =========================================================
# DETAILED DIAGNOSIS
# =========================================================

st.subheader("🔍 Diagnosis Summary")

if faults:

    for fault in faults:

        st.write(
            f"⚠️ {fault}"
        )

else:

    st.write(
        "✅ No significant abnormal fault indicators detected."
    )


st.info(
    f"""
    **Conveyor Status:** {diagnosis_status}

    **Overall Conveyor Health:** {overall_health:.1f}%

    **Failure-Risk Index:** {risk_index:.1f}%

    **Camera Detection:** {camera_result}

    **Belt Direction:** {camera_direction}

    **Vertical Visual Deviation:** {camera_deviation:.1f}%

    **Damage Severity:** {damage_severity:.1f}% — {damage_level}

    **Recommended Action:** Review the maintenance recommendations
    above and inspect the affected conveyor components if abnormal
    indicators persist.
    """
)


# =========================================================
# SYSTEM STATUS
# =========================================================

st.header("⚙️ SYSTEM STATUS")

s1, s2, s3, s4 = st.columns(4)

s1.metric(
    "Camera",
    "ONLINE"
)

s2.metric(
    "Sensor Fusion",
    "ACTIVE"
)

s3.metric(
    "FFT Analysis",
    "ACTIVE"
)

s4.metric(
    "Predictive Assessment",
    "ACTIVE"
)


# =========================================================
# TECHNICAL METHODOLOGY
# =========================================================

st.header("🧪 Technical Methodology")

with st.expander(
    "View technical implementation"
):

    st.markdown(
        """
        ### Data Acquisition

        The prototype uses four simulated physical monitoring channels controlled directly from the live input sliders:

        - Vibration
        - RPM
        - Motor Current
        - Load Cell

        An overhead camera channel is additionally simulated for belt
        tracking and vertical displacement detection.

        ### Signal Processing

        Vibration data is generated as a time-domain signal containing
        periodic components and noise. RMS analysis estimates vibration
        magnitude, while FFT transforms the signal into the frequency
        domain to identify dominant frequency components.

        ### Sensor Health

        Individual health scores are calculated from deviation from
        predefined operating references.

        The sensor health score uses:

        - Vibration = 35%
        - RPM = 20%
        - Motor Current = 25%
        - Load = 20%

        ### Multimodal Fusion

        The final conveyor health combines:

        - Physical sensor health = 80%
        - Camera visual health = 20%

        This produces a single simulated Conveyor Health Score.

        ### Camera Vertical Alignment

        The simulated camera establishes a calibrated horizontal
        reference centerline.

        The reference centerline remains fixed.

        Normal forward belt movement occurs along the conveyor travel
        direction and is not considered a misalignment.

        The system monitors the vertical position of the detected belt
        centerline relative to the fixed reference centerline.

        Both upward and downward displacement are considered for
        misalignment detection.

        - 0% to 10% vertical deviation → NORMAL
        - >10% to 20% → WARNING
        - >20% → CRITICAL

        Positive displacement indicates upward movement.

        Negative displacement indicates downward movement.

        ### Predictive Assessment

        The prototype converts the fused health score into a simulated
        Failure-Risk Index:

        **Failure Risk = 100 − Overall Health**

        Historical field data would be required to calibrate this into
        a statistically validated probability of failure.

        ### Fault Diagnosis

        Multiple abnormal indicators are combined to identify possible
        conditions such as:

        - Bearing / idler degradation
        - Belt slip
        - Motor overload
        - Vertical belt misalignment
        - Joint / belt damage
        """
    )


# =========================================================
# MECHANICAL INNOVATION
# =========================================================

st.header("🔩 Mechanical Innovation")

with st.expander(
    "View static-shaft idler concept"
):

    st.markdown(
        """
    ### Key Advantages

        - No slip rings
        - No rotating electrical connection
        - No moving sensor wires
        - Sensor positioned close to the idler/bearing source
        - Retrofit-oriented concept
        - Low-cost condition monitoring architecture

        The overhead camera provides a separate visual channel for belt
        tracking and vertical misalignment detection.
        """
    )


# =========================================================
# IMPORTANT DISCLAIMER
# =========================================================

st.warning(
    """
    **Prototype / Simulation Disclaimer**

    This application is a simulation-based proof of concept for SIH.
    Sensor values, vibration signals, camera frames and camera
    misalignment measurements are simulated.

    The camera module demonstrates the intended visual-detection workflow;
    it is not claiming real-time AI/YOLO/OpenCV detection in this version.

    The Failure-Risk Index is a simulated engineering indicator and is
    not a statistically validated probability of conveyor failure.

    Field deployment would require real sensor data, calibrated thresholds,
    historical failure datasets, industrial validation and integration
    with the actual conveyor control/maintenance system.
    """
)
