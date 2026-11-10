
import csi
import time

# ========== 1. 相机内参 ==========
FX = 289.62414929
FY = 289.61217747
CX = 140.26695932
CY = 117.88057319

# ========== 2. 畸变系数 ==========
K1 = -0.09478331
K2 =  0.24508433
P1 = -0.00266639
P2 =  0.00144091
K3 = -0.39222382

# ========== 3. 深度参数 ==========
# 相机光心到传感器顶部平面的光轴深度
TARGET_Z_MM = 50.0

# ========== 4. 圆形检测参数 ==========
R_MIN = 12
R_MAX = 100
THRESHOLD = 2500

# 连续10帧稳定性判断
WINDOW = 10
MIN_VALID = 8
MAX_MOVE = 5
MAX_R_CHANGE = 5


# ========== 5. 去畸变计算 ==========
def undistort_pixel(u, v):

    xd = (u - CX) / FX
    yd = (v - CY) / FY

    x = xd
    y = yd

    # 迭代反解畸变模型
    for i in range(6):

        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r4*r2

        radial = (
            1.0 + K1*r2 + K2*r4 + K3*r6
        )

        if abs(radial) < 1e-8:
            return xd, yd

        dx = (
            2*P1*x*y +
            P2*(r2 + 2*x*x)
        )

        dy = (
            P1*(r2 + 2*y*y) +
            2*P2*x*y
        )

        x = (xd - dx) / radial
        y = (yd - dy) / radial

    return x, y


# ========== 6. 像素转三维坐标 ==========
def pixel_to_camera(u, v, z_mm):

    xn, yn = undistort_pixel(u, v)

    X = xn * z_mm
    Y = yn * z_mm
    Z = z_mm

    return X, Y, Z


# ========== 7. 初始化 OpenMV ==========
cam = csi.CSI()
cam.reset()

cam.pixformat(csi.GRAYSCALE)
cam.framesize(csi.QVGA)

cam.snapshot(time=2000)
cam.auto_gain(False)

clock = time.clock()
history = []
frame_count = 0

print("Circle XYZ Test Start")


# ========== 8. 实时检测 ==========
while True:

    clock.tick()
    frame_count += 1

    img = cam.snapshot()

    circles = img.find_circles(
        threshold=THRESHOLD,
        x_margin=8,
        y_margin=8,
        r_margin=6,
        r_min=R_MIN,
        r_max=R_MAX,
        r_step=2
    )

    target = None

    if circles:

        target = max(
            circles,
            key=lambda c: c.magnitude
        )

        u = target.x
        v = target.y
        r = target.r
        mag = target.magnitude

        # 计算相机坐标（mm）
        X, Y, Z = pixel_to_camera(
            u, v, TARGET_Z_MM
        )

        history.append((u, v, r))

        # 绘制检测圆
        img.draw_circle(
            (u, v, r),
            color=(255, 255, 255),
            thickness=2
        )

        # 绘制圆心
        img.draw_cross(
            (u, v),
            color=(255, 255, 255),
            size=10
        )

        # 显示像素坐标
        img.draw_string(
            (5, 5),
            "PX:%d PY:%d R:%d" % (u, v, r),
            color=(255, 255, 255)
        )

        # 显示毫米坐标
        img.draw_string(
            (5, 25),
            "X: %.2f mm" % X,
            color=(255, 255, 255)
        )

        img.draw_string(
            (5, 45),
            "Y: %.2f mm" % Y,
            color=(255, 255, 255)
        )

        img.draw_string(
            (5, 65),
            "Z: %.2f mm" % Z,
            color=(255, 255, 255)
        )

    else:

        history.append(None)

        img.draw_string(
            (5, 5),
            "Target Not Found",
            color=(255, 255, 255)
        )

    # ========== 9. 稳定性判断 ==========
    if len(history) > WINDOW:
        history.pop(0)

    valid = [p for p in history if p is not None]

    stable = False

    if len(history) == WINDOW:
        if len(valid) >= MIN_VALID:

            xs = [p[0] for p in valid]
            ys = [p[1] for p in valid]
            rs = [p[2] for p in valid]

            dx = max(xs) - min(xs)
            dy = max(ys) - min(ys)
            dr = max(rs) - min(rs)

            if (dx <= MAX_MOVE and
                dy <= MAX_MOVE and
                dr <= MAX_R_CHANGE):
                stable = True

    if stable:
        status = "TARGET STABLE"
    else:
        status = "NOT READY"

    img.draw_string(
        (5, 85),
        status,
        color=(255, 255, 255)
    )

    # ========== 10. 串口终端输出 ==========
    if frame_count % 10 == 0:

        if target is not None:

            print("========== TARGET ==========")

            print(
                "Pixel: (%d, %d), R=%d"
                % (u, v, r)
            )

            print(
                "Camera XYZ: %.2f, %.2f, %.2f mm"
                % (X, Y, Z)
            )

            print("MAG:", mag)
            print("Stable:", stable)

        else:
            print("Target Not Found")
