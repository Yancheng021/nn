# --------------------------
# 简化修复版：确保车辆正确生成
# --------------------------

import carla
import time
import numpy as np
import cv2
import math
from collections import deque
import random


class TrafficLightDetector:
    """交通灯检测器 - 检测前方交通灯状态"""

    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.map = world.get_map()
        self.stopped_at_light = False  # 停车标志

    def should_stop(self):
        """检测是否应该停车（只在行驶方向所在车道的红灯时）"""
        location = self.vehicle.get_location()
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6  # km/h

        # 获取车辆当前道路的行驶方向（更精确）
        waypoint = self.map.get_waypoint(location, project_to_road=True)
        if waypoint:
            # 使用道路的行驶方向（考虑道路是单向还是双向）
            road_forward = waypoint.transform.get_forward_vector()
            road_lane_id = waypoint.road_id  # 获取道路ID
            road_lane = waypoint.lane_id  # 获取车道ID（区分左/右车道）
        else:
            # 如果找不到路点，使用车辆朝向
            road_forward = transform.get_forward_vector()
            road_lane_id = None
            road_lane = None

        # 获取车辆前方的交通灯
        lights = self.world.get_actors().filter('traffic.traffic_light*')

        if not lights:
            return False, float('inf')

        closest_red_light = None
        closest_distance = float('inf')

        # 检测前方的交通灯
        for light in lights:
            light_location = light.get_location()

            # 计算距离
            distance = location.distance(light_location)

            # 只检测40米范围内的交通灯
            if distance > 40.0:
                continue

            # 计算交通灯相对于车辆的位置
            to_light = carla.Vector3D(
                light_location.x - location.x,
                light_location.y - location.y,
                light_location.z - location.z
            )

            # 检查交通灯是否在车辆行驶方向的前方
            # 使用道路行驶方向而不是车辆朝向
            dot = road_forward.x * to_light.x + road_forward.y * to_light.y

            if dot > 0:  # 在行驶方向前方
                # 检查交通灯状态
                state = light.get_state()

                if state == carla.TrafficLightState.Red:
                    # 检查交通灯是否在同一个车道
                    light_waypoint = self.map.get_waypoint(light_location, project_to_road=True)
                    if light_waypoint and road_lane_id is not None:
                        # 只检测同一道路上的交通灯
                        if light_waypoint.road_id == road_lane_id:
                            if distance < closest_distance:
                                closest_distance = distance
                                closest_red_light = light
                    else:
                        # 如果无法获取交通灯所在道路，使用原逻辑
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_red_light = light
                elif state == carla.TrafficLightState.Yellow:
                    # 黄灯时根据距离决定是否停车
                    light_waypoint = self.map.get_waypoint(light_location, project_to_road=True)
                    if light_waypoint and road_lane_id is not None:
                        if light_waypoint.road_id == road_lane_id and distance < 15.0 and distance < closest_distance:
                            closest_distance = distance
                            closest_red_light = light
                    else:
                        if distance < 15.0 and distance < closest_distance:
                            closest_distance = distance
                            closest_red_light = light

        if closest_red_light:
            return True, closest_distance

        return False, float('inf')


class SimpleController:
    """简单但可靠的控制逻辑"""

    def __init__(self, world, vehicle, is_main_vehicle=False):
        self.world = world
        self.vehicle = vehicle
        self.map = world.get_map()
        self.target_speed = 50.0  # km/h，增加最高速度限制
        self.waypoint_distance = 5.0
        self.last_waypoint = None
        self.manual_reverse = False  # 手动倒车标志
        self.manual_mode = False  # 手动驾驶模式标志
        self.manual_throttle = 0.0  # 手动油门
        self.manual_steer = 0.0  # 手动转向
        self.manual_brake = 0.0  # 手动刹车
        self.is_main_vehicle = is_main_vehicle  # 是否是主车辆
        self.traffic_light_detector = TrafficLightDetector(world, vehicle)

    def get_control(self):
        """基于路点的简单控制（支持手动和自动模式）"""
        # 获取车辆状态
        location = self.vehicle.get_location()
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()

        # 计算速度（考虑倒车方向）
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6  # km/h

        # 手动驾驶模式：返回手动控制值
        if self.manual_mode:
            return self.manual_throttle, self.manual_brake, self.manual_steer, self.manual_reverse

        # 检查是否在倒车模式
        if self.manual_reverse:
            # 倒车模式：直接返回倒车控制
            return 0.3, 0.0, 0.0, True  # throttle, brake, steer, reverse

        # 获取路点
        waypoint = self.map.get_waypoint(location, project_to_road=True)

        if not waypoint:
            # 如果没有找到路点，返回保守控制
            # return 0.3, 0.0, 0.0  # 原返回值（3个值）
            return 0.3, 0.0, 0.0, False  # 新返回值（4个值，增加reverse标志）

        # 获取下一个路点
        next_waypoints = waypoint.next(self.waypoint_distance)

        if not next_waypoints:
            # 如果没有下一个路点，使用当前路点
            target_waypoint = waypoint
        else:
            target_waypoint = next_waypoints[0]

        self.last_waypoint = target_waypoint

        # 计算转向
        vehicle_yaw = math.radians(transform.rotation.yaw)
        target_loc = target_waypoint.transform.location

        # 计算相对位置
        dx = target_loc.x - location.x
        dy = target_loc.y - location.y

        local_x = dx * math.cos(vehicle_yaw) + dy * math.sin(vehicle_yaw)
        local_y = -dx * math.sin(vehicle_yaw) + dy * math.cos(vehicle_yaw)

        if abs(local_x) < 0.1:
            steer = 0.0
        else:
            angle = math.atan2(local_y, local_x)
            steer = max(-0.5, min(0.5, angle / 1.0))

        # 速度控制
        if speed < self.target_speed * 0.8:
            throttle, brake = 0.6, 0.0
        elif speed > self.target_speed * 1.2:
            throttle, brake = 0.0, 0.3
        else:
            throttle, brake = 0.3, 0.0

        # 交通灯检测（仅对主车辆生效）
        if self.is_main_vehicle:
            should_stop, distance = self.traffic_light_detector.should_stop()
            if should_stop and distance < 15.0:  # 只在距离交通灯15米内才开始停车
                if speed > 5.0:
                    # 需要减速
                    throttle = 0.0
                    brake = 0.5
                else:
                    # 已经停止
                    throttle = 0.0
                    brake = 1.0
                    if not self.traffic_light_detector.stopped_at_light:
                        print(f"红灯停车，距离交通灯 {distance:.1f} 米")
                        self.traffic_light_detector.stopped_at_light = True
            else:
                # 绿灯或无交通灯，或距离较远，重置停车标志
                self.traffic_light_detector.stopped_at_light = False

        # return throttle, brake, steer  # 原返回值（3个值）
        return throttle, brake, steer, False  # 新返回值（4个值，增加reverse标志）

    def toggle_reverse(self):
        """切换倒车模式"""
        self.manual_reverse = not self.manual_reverse
        if self.manual_reverse:
            print("进入倒车模式")
        else:
            print("退出倒车模式，恢复前进")


class SimpleDrivingSystem:
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicle = None
        self.vehicles = []  # 存储所有车辆（包括主车和NPC）
        self.controllers = []  # 存储每个车辆的控制器
        self.current_vehicle_index = 0  # 当前关注的车辆索引
        self.cameras = {}  # 存储多个相机
        self.controller = None
        self.camera_image = None
        self.night_mode = False  # 夜晚模式标志

    def set_weather(self, day=True):
        """设置天气"""
        if day:
            weather = carla.WeatherParameters(
                cloudiness=30.0,
                precipitation=0.0,
                sun_altitude_angle=70.0
            )
        else:
            weather = carla.WeatherParameters(
                cloudiness=0.0,
                precipitation=0.0,
                sun_altitude_angle=-90.0  # 夜晚
            )
        self.world.set_weather(weather)

    def toggle_night_mode(self):
        """切换夜晚模式"""
        self.night_mode = not self.night_mode
        self.set_weather(day=not self.night_mode)

        if self.night_mode:
            print("已切换到夜晚模式 - 已开启近光灯")
            self.set_headlights(True)
        else:
            print("已切换到白天模式 - 已关闭近光灯")
            self.set_headlights(False)

    def set_headlights(self, enabled):
        """设置所有车辆的近光灯"""
        for vehicle in self.vehicles:
            if vehicle is not None and vehicle.is_alive:
                try:
                    light_state = carla.VehicleLightState.LowBeam
                    if enabled:
                        vehicle.set_light_state(light_state)
                    else:
                        vehicle.set_light_state(carla.VehicleLightState.NONE)
                except:
                    pass

    def connect(self):
        """连接到CARLA服务器"""
        print("正在连接到CARLA服务器...")

        try:
            # 尝试多种连接方式
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)

            # 检查可用地图
            available_maps = self.client.get_available_maps()
            print(f"可用地图: {available_maps}")

            # 加载地图
            self.world = self.client.load_world('Town01')
            print("地图加载成功")

            # 设置同步模式
            settings = self.world.get_settings()
            settings.synchronous_mode = False  # 先使用异步模式确保连接
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

            print("连接成功！")
            return True

        except Exception as e:
            print(f"连接失败: {e}")
            print("请确保:")
            print("1. CARLA服务器正在运行")
            print("2. 服务器端口为2000")
            print("3. 地图Town01可用")
            return False

    def spawn_vehicle(self):
        """生成车辆 - 简化版本"""
        print("正在生成车辆...")

        try:
            # 获取蓝图库
            blueprint_library = self.world.get_blueprint_library()

            # 选择车辆蓝图
            vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
            if not vehicle_bp:
                print("未找到特斯拉蓝图，尝试其他车辆...")
                vehicle_bp = blueprint_library.filter('vehicle.*')[0]

            vehicle_bp.set_attribute('color', '255,0,0')  # 红色

            # 获取出生点
            spawn_points = self.world.get_map().get_spawn_points()
            print(f"找到 {len(spawn_points)} 个出生点")

            if not spawn_points:
                print("没有可用的出生点！")
                return False

            # 选择第一个出生点
            spawn_point = spawn_points[0]

            # 尝试生成车辆
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

            if not self.vehicle:
                print("无法生成车辆，尝试清理现有车辆...")
                # 清理现有车辆
                for actor in self.world.get_actors().filter('vehicle.*'):
                    actor.destroy()
                time.sleep(0.5)

                # 再次尝试
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

            if self.vehicle:
                print(f"车辆生成成功！ID: {self.vehicle.id}")
                print(f"位置: {spawn_point.location}")

                # 禁用自动驾驶
                self.vehicle.set_autopilot(False)

                return True
            else:
                print("车辆生成失败")
                return False

        except Exception as e:
            print(f"生成车辆时出错: {e}")
            return False

    def setup_camera(self):
        """设置多个相机"""
        print("正在设置相机...")

        try:
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')

            # 设置相机属性
            camera_bp.set_attribute('image_size_x', '640')
            camera_bp.set_attribute('image_size_y', '480')
            camera_bp.set_attribute('fov', '90')

            # 第一人称相机
            first_person_transform = carla.Transform(
                carla.Location(x=2.0, z=1.2),  # 驾驶座位置
                carla.Rotation(pitch=0.0)  # 平视
            )
            first_person_camera = self.world.spawn_actor(
                camera_bp, first_person_transform, attach_to=self.vehicle
            )
            first_person_camera.listen(lambda image: self.camera_callback(image, 'first_person'))
            self.cameras['first_person'] = first_person_camera

            # 第三人称相机
            third_person_transform = carla.Transform(
                carla.Location(x=-8.0, z=6.0),  # 在车辆后方上方
                carla.Rotation(pitch=-20.0)  # 向下看
            )
            third_person_camera = self.world.spawn_actor(
                camera_bp, third_person_transform, attach_to=self.vehicle
            )
            third_person_camera.listen(lambda image: self.camera_callback(image, 'third_person'))
            self.cameras['third_person'] = third_person_camera

            # 鸟瞰图相机
            birdseye_transform = carla.Transform(
                carla.Location(x=0.0, z=30.0),  # 车辆正上方30米
                carla.Rotation(pitch=-90.0)  # 垂直向下
            )
            birdseye_camera = self.world.spawn_actor(
                camera_bp, birdseye_transform, attach_to=self.vehicle
            )
            birdseye_camera.listen(lambda image: self.camera_callback(image, 'birdseye'))
            self.cameras['birdseye'] = birdseye_camera

            print("相机设置成功 - 已创建三个视角相机")
            return True

        except Exception as e:
            print(f"设置相机时出错: {e}")
            return False

    def camera_callback(self, image, view_mode=None):
        """相机数据回调"""
        try:
            # 只有当前视角的相机数据才会被使用
            if view_mode == self.current_view:
                # 转换图像数据
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4))
                self.camera_image = array[:, :, :3]  # RGB通道
        except:
            pass

    def update_camera_view(self):
        """更新相机视角"""
        print(f"已切换到{self.get_view_name()}视角")

    def get_view_name(self):
        """获取视角名称"""
        view_names = {
            'first_person': 'First Person',
            'third_person': 'Third Person',
            'birdseye': 'Birds Eye'
        }
        return view_names.get(self.current_view, 'Unknown')

    def setup_all_vehicles_cameras(self):
        """为所有车辆设置相机系统"""
        print("正在为所有车辆设置相机...")

        try:
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')

            # 设置相机属性
            camera_bp.set_attribute('image_size_x', '640')
            camera_bp.set_attribute('image_size_y', '480')
            camera_bp.set_attribute('fov', '90')

            # 为每辆车创建相机
            for i, vehicle in enumerate(self.vehicles):
                if vehicle is None:
                    continue

                print(f"为车辆 {i} 设置相机...")

                # 第三人称相机 - 用于跟随视角
                third_person_transform = carla.Transform(
                    carla.Location(x=-8.0, z=6.0),  # 在车辆后方上方
                    carla.Rotation(pitch=-20.0)  # 向下看
                )
                third_person_camera = self.world.spawn_actor(
                    camera_bp, third_person_transform, attach_to=vehicle
                )
                third_person_camera.listen(lambda image, idx=i: self.camera_callback(image, idx, 'third_person'))
                self.cameras[f'vehicle_{i}_third_person'] = third_person_camera

            print(f"已为 {len(self.vehicles)} 辆车设置相机")
            return True

        except Exception as e:
            print(f"设置多车辆相机时出错: {e}")
            return False

    def camera_callback(self, image, vehicle_index, view_mode=None):
        """相机数据回调"""
        try:
            # 只有当前关注车辆的相机数据才会被使用
            if vehicle_index == self.current_vehicle_index and view_mode == 'third_person':
                # 转换图像数据
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4))
                self.camera_image = array[:, :, :3]  # RGB通道
        except:
            pass

    def update_current_vehicle_view(self):
        """更新当前关注的车辆视角"""
        vehicle_label = "主车辆（红色特斯拉）" if self.current_vehicle_index == 0 else f"NPC车辆 {self.current_vehicle_index}"
        print(f"已切换到: {vehicle_label} 视角")

    def setup_controller(self):
        """设置控制器"""
        self.controller = SimpleController(self.world, self.vehicle, is_main_vehicle=True)
        print("控制器设置完成")

    def setup_all_controllers(self):
        """为所有车辆设置控制器"""
        print("正在为所有车辆设置控制器...")
        self.controllers = []

        for i, vehicle in enumerate(self.vehicles):
            controller = SimpleController(self.world, vehicle, is_main_vehicle=(i == 0))
            self.controllers.append(controller)
            vehicle_name = "主车辆" if i == 0 else f"NPC车辆{i}"
            print(f"  {vehicle_name} 控制器设置完成")

        # 确保self.controller指向主车辆控制器
        self.controller = self.controllers[0]
        print("所有控制器设置完成")

    def run(self):
        """主运行循环"""
        print("\n" + "=" * 50)
        print("简化自动驾驶系统")
        print("=" * 50)

        # 连接服务器
        if not self.connect():
            return

        # 生成车辆
        if not self.spawn_vehicle():
            return

        # 设置控制器
        self.setup_controller()

        # 等待一会儿让系统稳定
        print("系统初始化中...")
        time.sleep(2.0)

        # 设置天气（默认白天）
        self.set_weather(day=True)

        # 生成2辆NPC车辆（加上主车辆共3辆特斯拉）
        npc_vehicles = self.spawn_npc_vehicles(2)

        # 将所有车辆添加到列表中（主车辆在前，NPC在后）
        self.vehicles = [self.vehicle] + npc_vehicles

        # 为所有车辆设置控制器
        self.setup_all_controllers()

        print("\n系统准备就绪！")
        print(f"共 {len(self.vehicles)} 辆车可用")
        print("控制指令:")
        print("  q - 退出程序")
        print("  r - 重置当前车辆")
        print("  s - 紧急停止")
        print("  x - 切换倒车/前进模式（速度为0时生效）")
        print("  m - 切换手动/自动驾驶模式")
        print("  j/k - 手动驾驶控制（仅在手动模式下，前进/倒车）")
        print("  l - 切换夜晚/白天模式（夜晚时开启近光灯）")
        for i in range(len(self.vehicles)):
            if i == 0:
                print(f"  1 - 切换到主车辆视角（红色特斯拉）")
            else:
                print(f"  {i+1} - 切换到NPC车辆{i}视角")
        print("\n开始自动驾驶...\n")

        # 为所有车辆创建相机
        self.setup_all_vehicles_cameras()

        frame_count = 0
        running = True
        
        # 卡住检测变量
        stuck_frame_count = 0
        stuck_threshold = 500  # 连续500帧速度低于2km/h认为卡住

        try:
            while running:
                # 获取当前关注车辆的状态
                current_vehicle = self.vehicles[self.current_vehicle_index]
                velocity = current_vehicle.get_velocity()
                speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6

                # 检测主车辆是否卡住
                if self.current_vehicle_index == 0:
                    if speed < 2.0:  # 速度低于2km/h
                        stuck_frame_count += 1
                        if stuck_frame_count >= stuck_threshold:
                            print("检测到主车辆卡住，自动重置...")
                            self.reset_vehicle()
                            stuck_frame_count = 0
                    else:
                        stuck_frame_count = 0

                # 对所有车辆应用控制器
                # 主车辆（索引0）使用自定义控制器
                controller = self.controllers[0]
                throttle, brake, steer, reverse = controller.get_control()
                control = carla.VehicleControl(
                    throttle=float(throttle),
                    brake=float(brake),
                    steer=float(steer),
                    hand_brake=False,
                    reverse=reverse
                )
                self.vehicle.apply_control(control)

                # 更新显示
                if self.camera_image is not None:
                    display_img = self.camera_image.copy()

                    # 显示当前车辆编号（绿色）
                    cv2.putText(display_img, f"Vehicle: {self.current_vehicle_index + 1}",
                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)

                    # 添加状态信息（显示当前关注车辆的数据）
                    cv2.putText(display_img, f"Speed: {speed:.1f} km/h",
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 2)
                    cv2.putText(display_img, f"Throttle: {throttle:.2f}",
                                (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 2)
                    cv2.putText(display_img, f"Steer: {steer:.2f}",
                                (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 2)
                    cv2.putText(display_img, f"Frame: {frame_count}",
                                (20, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 2)

                    # 显示手动驾驶模式状态
                    if self.current_vehicle_index == 0 and self.controller.manual_mode:
                            cv2.putText(display_img, "MANUAL MODE",
                                        (20, 210), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (0, 255, 255), 2)

                    # 显示倒车状态
                    if self.current_vehicle_index == 0 and self.controller.manual_reverse:
                            cv2.putText(display_img, "REVERSE MODE",
                                        (20, 180), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (0, 0, 255), 2)

                    cv2.imshow('Autonomous Driving - Multi-Vehicle View', display_img)

                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("正在退出...")
                    running = False
                elif key == ord('r'):
                    self.reset_current_vehicle()
                elif key == ord('s'):
                    # 紧急停止（只对主车辆生效）
                    if self.current_vehicle_index == 0:
                        self.vehicle.apply_control(carla.VehicleControl(
                            throttle=0.0, brake=1.0, hand_brake=True
                        ))
                    print("紧急停止")
                elif key == ord('x'):
                    # 切换倒车模式（只在速度接近0时允许切换）
                    if self.current_vehicle_index == 0 and speed < 1.0:
                        self.controller.toggle_reverse()
                    elif self.current_vehicle_index != 0:
                        print("只有主车辆可以切换倒车模式")
                    else:
                        print("请先减速到接近停止（速度<1km/h）再切换倒车模式")
                elif key == ord('m') or key == ord('M'):
                    # 切换手动/自动驾驶模式（只对主车辆生效）
                    if self.current_vehicle_index == 0:
                        self.controller.manual_mode = not self.controller.manual_mode
                        if self.controller.manual_mode:
                            print("已切换到手动驾驶模式 - 使用J前进/K倒车")
                        else:
                            print("已切换到自动驾驶模式")
                    else:
                        print("只有主车辆可以切换手动/自动驾驶模式")
                elif key == ord('l') or key == ord('L'):
                    # 切换夜晚模式
                    self.toggle_night_mode()
                elif key == ord('j') or key == ord('J'):
                    # 手动前进（只对主车辆生效，且在手动模式下）
                    if self.current_vehicle_index == 0:
                        if self.controller.manual_mode:
                            self.controller.manual_throttle = min(1.0, self.controller.manual_throttle + 0.1)
                            self.controller.manual_reverse = False  # 前进模式
                elif key == ord('k') or key == ord('K'):
                    # 手动倒车（只对主车辆生效，且在手动模式下）
                    if self.current_vehicle_index == 0:
                        if self.controller.manual_mode:
                            self.controller.manual_throttle = min(0.5, self.controller.manual_throttle + 0.1)
                            self.controller.manual_reverse = True  # 倒车模式
                elif ord('1') <= key <= ord('9'):
                    # 动态切换车辆视角（按数字键1-9）
                    vehicle_index = key - ord('1')
                    if vehicle_index < len(self.vehicles):
                        self.current_vehicle_index = vehicle_index
                        self.update_current_vehicle_view()
                    else:
                        print(f"车辆 {vehicle_index + 1} 不存在")
                else:
                    # 松开按键时自动减速（只在手动模式下）
                    if self.current_vehicle_index == 0 and self.controller.manual_mode:
                        if self.controller.manual_throttle > 0:
                            self.controller.manual_throttle = max(0.0, self.controller.manual_throttle - 0.05)
                        elif self.controller.manual_throttle < 0:
                            self.controller.manual_throttle = min(0.0, self.controller.manual_throttle + 0.05)

                frame_count += 1

                # 每100帧显示一次状态
                if frame_count % 100 == 0:
                    print(f"运行中... 帧数: {frame_count}, 速度: {speed:.1f} km/h")

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"运行错误: {e}")
        finally:
            self.cleanup()

    def spawn_npc_vehicles(self, count=2):
        """生成NPC车辆（改进版：确保生成在合适位置）"""
        print(f"正在生成 {count} 辆NPC车辆...")

        try:
            blueprint_library = self.world.get_blueprint_library()
            spawn_points = self.world.get_map().get_spawn_points()

            # 获取主车辆当前位置
            main_location = self.vehicle.get_location()

            # 筛选出距离主车辆较远的出生点（至少100米）
            far_spawn_points = [
                sp for sp in spawn_points
                if sp.location.distance(main_location) > 100.0
            ]

            # 如果找到的出生点不够，使用间隔更远的
            if len(far_spawn_points) < count:
                # 重新筛选，使用间隔至少20米的出生点
                far_spawn_points = []
                for sp in spawn_points:
                    if sp.location.distance(main_location) > 60.0:
                        # 检查是否与已选出生点间隔足够
                        too_close = False
                        for existing_sp in far_spawn_points:
                            if sp.location.distance(existing_sp.location) < 20.0:
                                too_close = True
                                break
                        if not too_close:
                            far_spawn_points.append(sp)

            if not far_spawn_points:
                far_spawn_points = spawn_points

            npc_vehicles = []

            # 生成指定数量的特斯拉车辆
            for i in range(count):
                # 使用间隔更大的出生点
                spawn_index = i * 5 % len(far_spawn_points)

                try:
                    # 使用特斯拉Model3蓝图
                    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
                    if not vehicle_bp:
                        print("未找到特斯拉蓝图，尝试其他车辆...")
                        vehicle_bp = blueprint_library.filter('vehicle.*')[0]

                    # 设置不同颜色区分
                    colors = [[0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]
                    color = colors[i % len(colors)]
                    vehicle_bp.set_attribute('color', f'{color[0]},{color[1]},{color[2]}')

                    # 生成NPC
                    npc = self.world.try_spawn_actor(vehicle_bp, far_spawn_points[spawn_index])

                    if npc:
                        # 立即开启自动驾驶，禁用交通灯检测
                        npc.set_autopilot(True, 16)
                        npc_vehicles.append(npc)
                        print(f"生成NPC车辆 {len(npc_vehicles)} (特斯拉Model3)")
                        print(f"  位置: {far_spawn_points[spawn_index].location}")
                    else:
                        print(f"无法在出生点 {spawn_index} 生成NPC车辆")
                except Exception as e:
                    print(f"生成NPC车辆 {i+1} 时出错: {e}")
                    pass

            print(f"成功生成 {len(npc_vehicles)} 辆NPC车辆")
            return npc_vehicles

        except Exception as e:
            print(f"生成NPC车辆时出错: {e}")
            return []

    def reset_vehicle(self):
        """重置车辆位置"""
        print("重置车辆...")

        spawn_points = self.world.get_map().get_spawn_points()
        if spawn_points:
            # 选择一个距离当前车辆较远的出生点，避免重置后立即撞到
            current_location = self.vehicle.get_location()
            far_spawn_points = [
                sp for sp in spawn_points
                if sp.location.distance(current_location) > 50.0  # 至少50米远
            ]
            if far_spawn_points:
                new_spawn_point = random.choice(far_spawn_points)
            else:
                new_spawn_point = random.choice(spawn_points)

            self.vehicle.set_transform(new_spawn_point)
            print(f"车辆已重置到新位置: {new_spawn_point.location}")

            # 重置控制器状态
            self.controllers[0].last_waypoint = None

            # 等待重置完成
            time.sleep(0.5)

    def reset_current_vehicle(self):
        """重置当前关注的车辆位置"""
        current_vehicle = self.vehicles[self.current_vehicle_index]
        vehicle_label = "主车辆" if self.current_vehicle_index == 0 else f"NPC车辆 {self.current_vehicle_index}"
        print(f"重置{vehicle_label}位置...")

        spawn_points = self.world.get_map().get_spawn_points()
        if spawn_points:
            # 选择一个距离当前位置较远的出生点
            current_location = current_vehicle.get_location()
            far_spawn_points = [
                sp for sp in spawn_points
                if sp.location.distance(current_location) > 50.0
            ]
            if far_spawn_points:
                new_spawn_point = random.choice(far_spawn_points)
            else:
                new_spawn_point = random.choice(spawn_points)

            current_vehicle.set_transform(new_spawn_point)
            print(f"{vehicle_label}已重置到新位置: {new_spawn_point.location}")

            # 如果是主车辆，重置控制器状态
            if self.current_vehicle_index == 0:
                self.controllers[0].last_waypoint = None
            # 如果是NPC车辆，重新设置自动驾驶
            else:
                current_vehicle.set_autopilot(True, 16)

            # 等待重置完成
            time.sleep(0.5)

    def cleanup(self):
        """清理资源"""
        print("\n正在清理资源...")

        # 清理所有相机
        for view_mode, camera in self.cameras.items():
            if camera:
                try:
                    camera.stop()
                    camera.destroy()
                except:
                    pass

        if self.vehicle:
            try:
                self.vehicle.destroy()
            except:
                pass

        # 等待销毁完成
        time.sleep(1.0)

        cv2.destroyAllWindows()
        print("清理完成")


def main():
    """主函数"""
    print("自动驾驶系统 - 简化版本")
    print("确保CARLA服务器正在运行...")

    system = SimpleDrivingSystem()
    system.run()


if __name__ == "__main__":
    main()