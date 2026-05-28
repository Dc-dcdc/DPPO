        touch_left_gripper = False
        touch_right_gripper = False
        # 存放当前帧里所有碰到针的物体名字
        objects_touching_needle = set()

        # 遍历底层物理引擎计算出的所有有效接触点 (Contacts)
        for i_contact in range(self._physics.data.ncon):
            # 提取发生碰撞的两个物体的底层 ID
            id_geom_1 = self._physics.data.contact[i_contact].geom1
            id_geom_2 = self._physics.data.contact[i_contact].geom2
            
            # 将底层 ID 翻译成你在 XML 环境文件中定义的名字 (如 "peg", "right_finger_1")
            geom1 = self._physics.model.id2name(id_geom_1, 'geom')
            geom2 = self._physics.model.id2name(id_geom_2, 'geom')
            
            # 只有两个物体都有名字时，才加入判定列表(双向记录)
            if geom1 and geom2:
                # 2. 只要有一方是针，就把另一方的名字扔进篮子
                if geom1 == "needle":
                    objects_touching_needle.add(geom2)
                elif geom2 == "needle":
                    objects_touching_needle.add(geom1)

        # 检查左夹爪是否成功捏住：左半边手指碰到了，且右半边手指也碰到了left_left中包含主模型right_right_finger和g0、g1、g2
        touched_left_left = any(obj.startswith("left_left") for obj in objects_touching_needle)
        touched_left_right = any(obj.startswith("left_right") for obj in objects_touching_needle)
        if touched_left_left and touched_left_right:
            touch_left_gripper = True
            
        # 检查右夹爪是否成功捏住
        touched_right_left = any(obj.startswith("right_left") for obj in objects_touching_needle)
        touched_right_right = any(obj.startswith("right_right") for obj in objects_touching_needle)
        if touched_right_left and touched_right_right:
            touch_right_gripper = True

        # ==========================================
        # 提取相关坐标
        # ==========================================
        needle_head = self._physics.named.data.geom_xpos['needle_head']                #针头
        needle_tail = self._physics.named.data.geom_xpos['needle_tail']                #针尾
        needle_left_pos = self._physics.named.data.geom_xpos['needle_mark_1_4']        #左臂抓取标记点 1/4处
        needle_right_pos = self._physics.named.data.geom_xpos['needle_mark_3_4']       #右臂抓取标记点 3/4处
        
        left_left_finger = self._physics.named.data.geom_xpos['left_left_g2']
        left_right_finger = self._physics.named.data.geom_xpos['left_right_g2']        
        right_left_finger = self._physics.named.data.geom_xpos['right_left_g2']        #右臂左指尖
        right_right_finger = self._physics.named.data.geom_xpos['right_right_g2']      #右臂右指尖

        hole_entrance = self._physics.named.data.geom_xpos['hole_entrance']            #出洞口
        hole_exit = self._physics.named.data.geom_xpos['hole_exit']                    #出洞口

        # ==========================================
        # 计算左右臂和针距离奖励
        # ==========================================
        # 计算右臂夹爪的中心点 (Pinch Center)
        left_gripper_center = (left_left_finger + left_right_finger) / 2.0
        right_gripper_center = (right_left_finger + right_right_finger) / 2.0

        # 计算右臂夹爪中心到针标记点的距离
        dist_left_to_mark = np.linalg.norm(left_gripper_center - needle_left_pos)
        dist_right_to_mark = np.linalg.norm(right_gripper_center - needle_right_pos)
        
        # ==========================================
        # 计算针穿过墙洞的奖励
        # ==========================================
        # 计算针头到入口的距离
        dist_head_to_entrance = np.linalg.norm(needle_head - hole_entrance)

        # 计算针头到“洞口出口(hole_exit)”的距离
        dist_head_to_exit = np.linalg.norm(needle_head - hole_exit)
        
        # 计算针尾到出口的距离
        dist_tail_to_exit = np.linalg.norm(needle_tail - hole_exit)

        # ==========================================
        # 阶段流转奖励逻辑
        # ==========================================
        # 每走一步扣 0.5 分，降低步数
        reward = -1

        # 触发 1：针头到达入口 (进洞),给个很小的值防止在外面就判断进去了 ，加上3d距离防止绕墙
        if needle_head[0] - hole_entrance[0] < 0.001 and dist_head_to_entrance < 0.01:
            if not self.needle_start_through: # 确保只触发一次
                self.needle_start_through = True
                reward += 25.0  # 突破瞬间的巨额奖励

        # 触发 2：针头到达出口 (露头)
        if needle_head[0] < hole_exit[0] and dist_head_to_exit < 0.02:
            if not self.needle_reached_exit:
                self.needle_reached_exit = True
                reward += 50.0  # 突破瞬间的巨额奖励

        # 触发 3：左手成功接应
        if self.needle_reached_exit and touch_left_gripper:
            if not self.left_has_grasped:
                self.left_has_grasped = True
                reward += 75.0  # 突破瞬间的巨额奖励
        
        # 触发 4：针尾完全拔出
        # # 穿孔方向为沿着 X 轴向负方向移动，针尾小于出口 且 距离小于设定值（防止从墙外面绕过）
        if needle_tail[0] < hole_exit[0] and dist_tail_to_exit < 0.02:
            if not self.needle_completely_through:
                self.needle_completely_through = True
                reward += 100.0  # 突破瞬间的巨额奖励
        
        if not self.left_has_grasped:
            # --- 前半场：右臂主导 ---

            if not touch_right_gripper:  # 右臂未抓针
                # 引导右臂接近 针3/4 处的奖励
                reward += 2.0 * np.exp(-15.0 * dist_right_to_mark)
            else: 
                reward += 0.25 # 保持抓取的微弱奖励
                if not self.needle_start_through:
                    # 没到达洞口，引导针头到达入口 （最高5分）
                    reward += 5 * np.exp(-15.0 * dist_head_to_entrance)
                elif not self.needle_reached_exit:
                    # 到达洞口，引导针头到达出口
                    reward += 5 * np.exp(-20.0 * dist_head_to_exit)
                else:
                    # 到达出口，引导左臂接近针标记
                    reward += 3 * np.exp(-15.0 * dist_left_to_mark)

        else: 
            # --- 后半场：左臂主导 ---
            reward += 0.25 
            # 惩罚右手，逼迫其松开并让开空间
            if touch_right_gripper:
                reward -= 0.5 

            if not touch_left_gripper:
                # 左手只是碰了一下但没抓稳，或者中途脱手了
                # 掉落惩罚，但因为有之前的一次性奖励撑腰，它依然敢于尝试交接
                # reward -= 0.5
                # 重新提供一个引导左臂去抓针的低保底引力，逼迫它捏紧双指

                reward += 3.0 * np.exp(-15.0 * dist_left_to_mark)
                
            else:
                reward += 0.25 # 保持抓紧的微弱奖励
                # 3. 终极判定：针尾是否完全越过出口

                if not self.needle_completely_through:
                    # 左手往外拔出
                    # 连续奖励：引导左臂继续往外拉，直到针尾到达出口
                    reward += 5.0 * np.exp(-15.0 * dist_tail_to_exit)
                else:
                    # print("✅ 针尾完全越过出口！")
                    # 针已完全拔出，执行抬举动作验证
                    wall_center_x = (hole_entrance[0] + hole_exit[0]) / 2.0
                    target_z_target = hole_exit[2] + 0.12 # 最低要求抬起 12 厘米，总高度17cm
                    
                    # 2. 计算针的中心点
                    needle_center = (needle_head + needle_tail) / 2.0
                    
                    # 3. 独立计算各轴误差 (解耦 Z 轴)
                    # Z轴误差：巧妙使用 max()，只要当前高度超过 target_z_min，误差就归零！举多高都不会受罚。
                    z_error = max(0.0, target_z_target - needle_center[2])
                    x_error = abs(needle_center[0] - wall_center_x)
                    y_error = abs(needle_center[1] - hole_exit[1])
                    
                    # 将解耦后的误差重新合成为复合距离，专供连续奖励计算使用
                    composite_error_dist = np.sqrt(x_error**2 + y_error**2 + z_error**2)
                    
                    # 4. 独立且宽容的成功判定条件
                    # 只要高度超过下限 (z_error == 0)，且 XY 平面没有偏离墙体中心太远 (比如 2 厘米内容差)
                    is_above_wall = (z_error == 0.0) and (x_error < 0.01) #and (y_error < 0.02)
                    
                    if not is_above_wall:
                        # 连续奖励：此时距离变量变成了 composite_error_dist
                        reward += 10.0 * np.exp(-15.0 * composite_error_dist)
                    else:
                        # 阶段 6：终极胜利！针被成功拔出、抬高，并稳稳处于墙的正上方
                        reward += 500.0 
                        self.terminated = True

        return float(reward)