import pyzed.sl as sl
import numpy as np

def test_zed_camera():
    # 1. 创建 ZED 相机对象
    zed = sl.Camera()

    # 2. 设置初始化参数
    init_params = sl.InitParameters()
    # 推荐在代码里明确请求深度模式，因为我们需要深度信息
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE 
    # 使用 720p 分辨率，帧率 60 (AV-ALOHA 论文中的常用设置)
    init_params.camera_resolution = sl.RESOLUTION.HD720 
    init_params.camera_fps = 60

    # 3. 尝试打开相机
    print("正在打开 ZED 相机，这可能需要几秒钟（初始化 CUDA）...")
    err = zed.open(init_params)
    
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"\n❌ 打开相机失败: {repr(err)}")
        print("请检查：1. USB 线是否插紧且连接到 USB 3.0 接口；2. 重启电脑。")
        return

    print("\n✅ ZED 相机打开成功！")
    
    # 4. 获取相机的内参 (测试通讯是否正常)
    camera_info = zed.get_camera_information()
    calibration_params = camera_info.camera_configuration.calibration_parameters
    left_cam = calibration_params.left_cam
    print(f"-> 相机分辨率: {left_cam.image_size.width} x {left_cam.image_size.height}")
    print(f"-> 左眼焦距 (fx, fy): ({left_cam.fx:.2f}, {left_cam.fy:.2f})")

    # 5. 准备对象来存储图像和深度图
    image_zed = sl.Mat()
    depth_zed = sl.Mat()

    # 6. 尝试抓取一帧图像
    print("\n正在尝试抓取图像和深度图...")
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # 获取左眼的 RGB 图像
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        # 获取深度图 (MEASURE_DEPTH 返回的是物理距离矩阵)
        zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
        
        # 将 sl.Mat 转换为 numpy 数组以便后续处理
        image_np = image_zed.get_data()
        depth_np = depth_zed.get_data()
        
        print("✅ 抓取成功！")
        print(f"-> RGB 图像 Numpy Shape: {image_np.shape} (H, W, 4)") # ZED 返回 BGRA
        print(f"-> 深度图 Numpy Shape: {depth_np.shape} (H, W)")
        
        # 打印画面中心的深度值 (测试深度计算是否正常)
        center_y, center_x = depth_np.shape[0] // 2, depth_np.shape[1] // 2
        center_distance = depth_np[center_y, center_x]
        print(f"-> 画面中心的物理距离: {center_distance:.3f} 米")
    else:
        print("❌ 抓取图像失败！")

    # 7. 关闭相机释放资源
    zed.close()
    print("\n相机已关闭，测试完成。")

if __name__ == "__main__":
    test_zed_camera()