using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using UnityEngine.XR;

//MonoBehaviour类，可以让脚本可以挂载到 GameObject 上，并且在 Unity 的生命周期内调用 Start、Update 等方法。
//GameObject是 Unity 中的基本对象，它真正有什么功能，取决于它身上挂了哪些组件 Component
public class Quest3PoseUdpSender : MonoBehaviour
{
    // 在 Unity Inspector 面板里暴露可调参数
    [Header("UDP Target")]                           // 在 Unity Inspector 面板里加一个标题分组 
    [Tooltip("电脑的局域网 IP，例如 192.168.1.100。")]  // 给下一个变量添加鼠标悬停提示
    public string receiverIp = "172.19.33.18";       // public字段，Unity 会自动把它们显示在 Inspector 面板里，可以输入、勾选、拖拽等方式设置它们的值

    [Tooltip("Python quest3_receiver.py 监听的端口。")]
    public int receiverPort = 5005;

    [Header("Streaming")]
    [Tooltip("发送频率，建议 30-90 Hz。")]
    public float sendHz = 60f;

    [Tooltip("是否在 Unity Console 中打印发送状态。")]
    public bool logPackets = false;

    [Header("Optional XR Rig Transforms")]
    [Tooltip("可选：拖入 XR Origin 里的 Main Camera。若绑定，会优先读取这个 Transform 的姿态。")]
    public Transform headTransform;                 // Transform 是 Unity 中可以在 Inspector 里把场景中的对象拖到这个变量上进行绑定的组件，代表位置、旋转、缩放等信息。

    [Tooltip("可选：拖入 XR Origin 里的 Left Controller。若绑定，会优先读取这个 Transform 的姿态。")]
    public Transform leftHandTransform;

    [Tooltip("可选：拖入 XR Origin 里的 Right Controller。若绑定，会优先读取这个 Transform 的姿态。")]
    public Transform rightHandTransform;

    [Tooltip("勾选后优先使用上面绑定的 Transform；未绑定时仍会回退到 XRNode。")]
    public bool preferSceneTransforms = true;

    [Tooltip("读取 Transform.localPosition/localRotation。若关闭，则读取世界坐标 position/rotation。")]
    public bool useLocalPose = true;

    private UdpClient udpClient;
    private IPEndPoint receiverEndPoint;
    private float lastSendTime;
    private int sequence;

    [Serializable] // 这个特性让这个类可以被 Unity 的 JsonUtility 序列化成 JSON 字符串，或者从 JSON 字符串反序列化回来。
    public class DevicePosePacket
    {
        public float[] pos = new float[3];    //定义数组来存储位置和旋转数据，因为 JSON 不直接支持 Vector3 和 Quaternion 类型，所以我们把它们转换成 float 数组来存储。
        public float[] quat = new float[4];
        public float index_trigger;           // 设备上的扳机值，通常是一个 0-1 之间的浮点数，表示扳机被按下的程度。
        public float hand_trigger;            // 设备上的握把值，通常也是一个 0-1 之间的浮点数，表示握把被按下的程度。
        public float thumbstick_x;            // 设备上的摇杆 X 轴值，通常是一个 -1 到 1 之间的浮点数，表示摇杆在水平方向上的偏移程度。
        public float thumbstick_y;            // 设备上的摇杆 Y 轴值，通常是一个 -1 到 1 之间的浮点数，表示摇杆在垂直方向上的偏移程度。
        public bool button_one;               // 设备上的主按钮状态A/X，表示主按钮是否被按下。
        public bool button_two;               // 设备上的次按钮状态B/Y，表示次按钮是否被按下。
        public bool button_thumbstick;        // 设备上的摇杆按钮状态，表示摇杆按钮是否被按下。
        public bool is_tracked;               // 设备是否被追踪。
        public uint tracking_state;           // 设备追踪状态。
        public bool device_valid;             // 设备是否有效。
        public bool position_valid;           // 位置数据是否有效。
        public bool rotation_valid;           // 旋转数据是否有效。
        public bool used_transform;           // 是否使用了绑定的 Transform。
        public string device_name;
    }

    [Serializable]                      //表示这个类可以被序列化，通常用于转成 JSON 发送
    public class Quest3Packet           // 这个类是我们最终要发送的数据包格式，包含一个序列号、一个时间戳，以及头部、左手、右手的位姿和控制信息。
    {
        public int sequence;            // 序列号，每发送一个包就加一，可以用来检测丢包和数据顺序。
        public double timestamp;        // 时间戳，记录发送这个包的时间，单位是秒，可以用来同步数据或者分析时序。
        public DevicePosePacket head;   // 头部的位姿和控制信息，使用上面定义的 DevicePosePacket 类来存储。
        public DevicePosePacket left;   // 左手的位姿和控制信息
        public DevicePosePacket right;  // 右手的位姿和控制信息
    }

    // Start 方法在 Unity 场景开始时被调用一次，我们在这里初始化 UDP 客户端和目标地址。
    private void Start()
    {
        udpClient = new UdpClient();   // 创建一个新的 UDP 客户端实例，用于发送数据包。
        receiverEndPoint = new IPEndPoint(IPAddress.Parse(receiverIp), receiverPort);  // 创建一个新的 IP 端点，使用上面设置的接收者 IP 和端口，这个端点表示我们要把数据发送到哪里。
        Debug.Log($"Quest3PoseUdpSender -> {receiverIp}:{receiverPort}");  // 在 Unity 的 Console 面板里打印一条日志，显示我们要发送数据的目标地址，方便调试和确认设置是否正确。
    }

    private void Update()
    {   
        // 这个条件判断用来控制发送频率，确保我们不会发送超过设置的 sendHz 的数据包。
        // Time.unscaledTime 是 Unity 提供的一个时间变量，表示从游戏开始到现在的时间，单位是秒，不受游戏时间缩放的影响。
        // 我们用它来计算距离上次发送数据包已经过去了多少时间，如果还没有达到 1/sendHz 秒，就返回不发送，这样就能控制发送频率了。
        if (sendHz > 0f && Time.unscaledTime - lastSendTime < 1f / sendHz) 
        {
            return;
        }

        lastSendTime = Time.unscaledTime;
        SendCurrentPose(); // 发送当前位姿和控制信息
    }

    private void SendCurrentPose()
    {
        // 创建一个新的 Quest3Packet 实例，填充序列号、时间戳，以及头部、左手、右手的位姿和控制信息。
        Quest3Packet packet = new Quest3Packet
        {
            sequence = sequence++,
            timestamp = Time.realtimeSinceStartupAsDouble,
            head = ReadNode(XRNode.Head, headTransform),             //读取头部的位姿和控制信息
            left = ReadNode(XRNode.LeftHand, leftHandTransform),     //读取左手的位姿和控制信息
            right = ReadNode(XRNode.RightHand, rightHandTransform),  //读取右手的位姿和控制信息
        };

        string json = JsonUtility.ToJson(packet);     // 将数据包转换为 JSON 字符串
        byte[] bytes = Encoding.UTF8.GetBytes(json);  // 将 JSON 字符串编码成 UTF-8 字节数组，准备发送
        udpClient.Send(bytes, bytes.Length, receiverEndPoint); // 使用 UDP 客户端发送字节数组到目标地址

        if (logPackets)
        {
            Debug.Log(json);
        }
    }

    // 创建ReadNode方法：先根据 XRNode 找到设备，再读取这个设备的位姿和控制信息，最后打包成 DevicePosePacket 返回。
    private DevicePosePacket ReadNode(XRNode node, Transform poseTransform)
    {
        InputDevice device = InputDevices.GetDeviceAtXRNode(node);  //通过node获取对应的设备，例如头部、左手、右手等XR设备

        Vector3 pos = Vector3.zero;
        Quaternion rot = Quaternion.identity;
        float indexTrigger = 0f;
        float handTrigger = 0f;
        Vector2 thumbstick = Vector2.zero;
        bool buttonOne = false;
        bool buttonTwo = false;
        bool thumbstickButton = false;
        bool isTracked = false;
        InputTrackingState trackingState = InputTrackingState.None;
        bool positionValid = false;
        bool rotationValid = false;
        bool usedTransform = false;

        if (preferSceneTransforms && poseTransform != null) // 优先使用绑定的 Transform，如果有的话
        {   
            // 根据 useLocalPose 选择读取 Transform.localPosition/localRotation。若关闭，则读取世界坐标 position/rotation
            pos = useLocalPose ? poseTransform.localPosition : poseTransform.position;
            rot = useLocalPose ? poseTransform.localRotation : poseTransform.rotation;
            usedTransform = true;
            positionValid = true;
            rotationValid = true;
            isTracked = true;
            trackingState = InputTrackingState.Position | InputTrackingState.Rotation;
        }

        if (!usedTransform) // 如果没有使用 Transform，就回退到使用 XRNode 设备的位姿数据
        {
            // TryGetFeatureValue 方法会尝试从设备上读取指定的特征值（例如位置、旋转、按钮状态等），
            // 如果成功读取到有效数据，就返回 true，并把数据存储在 out 参数里；如果设备没有这个特征，或者当前无法提供有效数据，就返回 false。
            positionValid = device.TryGetFeatureValue(CommonUsages.devicePosition, out pos);  
            rotationValid = device.TryGetFeatureValue(CommonUsages.deviceRotation, out rot);
        }
        device.TryGetFeatureValue(CommonUsages.trigger, out indexTrigger);
        device.TryGetFeatureValue(CommonUsages.grip, out handTrigger);
        device.TryGetFeatureValue(CommonUsages.primary2DAxis, out thumbstick);
        device.TryGetFeatureValue(CommonUsages.primaryButton, out buttonOne);
        device.TryGetFeatureValue(CommonUsages.secondaryButton, out buttonTwo);
        device.TryGetFeatureValue(CommonUsages.primary2DAxisClick, out thumbstickButton);
        device.TryGetFeatureValue(CommonUsages.isTracked, out isTracked);
        device.TryGetFeatureValue(CommonUsages.trackingState, out trackingState);

        return new DevicePosePacket
        {
            pos = new[] { pos.x, pos.y, pos.z },            // 把 Vector3 和 Quaternion 转换成 float 数组，方便序列化成 JSON
            quat = new[] { rot.x, rot.y, rot.z, rot.w },
            index_trigger = indexTrigger,
            hand_trigger = handTrigger,
            thumbstick_x = thumbstick.x,
            thumbstick_y = thumbstick.y,
            button_one = buttonOne,
            button_two = buttonTwo,
            button_thumbstick = thumbstickButton,
            is_tracked = isTracked,
            tracking_state = (uint)trackingState,
            device_valid = device.isValid,
            position_valid = positionValid,                 // 设备是否提供了有效的位置数据
            rotation_valid = rotationValid,                 // 设备是否提供了有效的旋转数据     
            used_transform = usedTransform,                 // 是否使用了绑定的 Transform 组件的数据，而不是 XRNode 设备的数据
            device_name = device.isValid ? device.name : "",
        };
    }

    private void OnDestroy()
    {
        udpClient?.Close();
        udpClient = null;
    }

    private void OnApplicationQuit()
    {
        udpClient?.Close();
        udpClient = null;
    }
}
