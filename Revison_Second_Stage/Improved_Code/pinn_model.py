import tensorflow as tf

class TwoTankPINN(tf.keras.Model):
    def __init__(self, neurons=56):
        super(TwoTankPINN, self).__init__()
        glorot_init = tf.keras.initializers.GlorotNormal()

        def scaled_he_init(shape, dtype=None):
            return 0.5 * tf.keras.initializers.HeNormal()(shape, dtype=dtype)

        # layer 1 outflow branch
        self.f1_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f1_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f1_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f1_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        # layer 2 outflow branch
        self.f2_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f2_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f2_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f2_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        # layer 3 outflow branch
        self.f3_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f3_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f3_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f3_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        # layer 4 outflow branch
        self.f4_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f4_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f4_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f4_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        # layer 5 outflow branch
        self.f5_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f5_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f5_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f5_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        self.f6_dense1 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=True)
        self.f6_dense2 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=True)
        self.f6_dense3 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=True)
        self.f6_out = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=scaled_he_init, use_bias=True)
        #################
        # Inter-tank flow branch
        self.f12_dense1 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f12_dense2 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f12_dense3 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f12_out = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=glorot_init, use_bias=False)

        self.f23_dense1 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f23_dense2 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f23_dense3 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f23_out = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=glorot_init, use_bias=False)

        self.f34_dense1 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f34_dense2 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f34_dense3 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f34_out = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=glorot_init, use_bias=False)

        self.f45_dense1 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f45_dense2 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f45_dense3 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f45_out = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=glorot_init, use_bias=False)

        self.f56_dense1 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f56_dense2 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f56_dense3 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f56_out = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=glorot_init, use_bias=False)


        # Trainable inflow scaler
        # Irrigation branch
        self.inflow_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.inflow_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.inflow_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.inflow_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        # Trainable inflow scaler
        # rain branch
        self.rain1_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain1_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain1_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain1_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        self.rain2_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain2_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain2_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain2_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        self.rain3_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain3_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain3_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain3_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        self.rain4_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain4_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain4_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain4_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        self.rain5_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain5_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain5_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain5_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        self.rain6_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain6_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain6_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain6_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        # plant_coef_1
        self.pc1_dense1 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc1_dense2 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc1_dense3 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc1_dense4 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc1_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=glorot_init, use_bias=False)

        # plant_coef_2
        self.pc2_dense1 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc2_dense2 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc2_dense3 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc2_dense4 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc2_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=glorot_init, use_bias=False)

        self.pc3_dense1 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc3_dense2 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc3_dense3 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc3_dense4 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc3_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=glorot_init, use_bias=False)

        self.pc4_dense1 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc4_dense2 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc4_dense3 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc4_dense4 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc4_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=glorot_init, use_bias=False)

        self.pc5_dense1 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc5_dense2 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc5_dense3 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc5_dense4 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc5_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=glorot_init, use_bias=False)

        self.pc6_dense1 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc6_dense2 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc6_dense3 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc6_dense4 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc6_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=glorot_init, use_bias=False)


        # Lateral outflow branches
        self.fl1_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl1_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl1_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl1_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        self.fl2_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl2_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl2_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl2_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        self.fl3_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl3_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl3_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl3_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        self.fl4_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl4_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl4_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl4_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        self.fl5_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl5_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl5_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl5_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

       

    def call(self, m, input_vec, t):

        # Unpack states
        m1 = tf.reshape(m[:, 0], [-1, 1])
        m2 = tf.reshape(m[:, 1], [-1, 1])
        m3 = tf.reshape(m[:, 2], [-1, 1])
        m4 = tf.reshape(m[:, 3], [-1, 1])
        m5 = tf.reshape(m[:, 4], [-1, 1])
        m6 = tf.reshape(m[:, 5], [-1, 1])

        # Compute differences between layers
        dm12 = m1 - m2
        dm23 = m2 - m3
        dm34 = m3 - m4
        dm45 = m4 - m5
        dm56 = m5 - m6

        # Input unpack
        qi   = tf.reshape(input_vec[:, 0], [-1, 1])  # irrigation
        rain = tf.reshape(input_vec[:, 1], [-1, 1])  # rain
        et0  = tf.reshape(input_vec[:, 2], [-1, 1])  # ET0
        t = tf.reshape(t, [-1, 1])  # time

        m1 = tf.identity(m1, name='m1')
        m2 = tf.identity(m2, name='m2')
        m3 = tf.identity(m3, name='m3')
        m4 = tf.identity(m4, name='m4')
        m5 = tf.identity(m5, name='m5')
        m6 = tf.identity(m6, name='m6')
        dm12 = tf.identity(dm12, name='dm12')
        dm23 = tf.identity(dm23, name='dm23')
        dm34 = tf.identity(dm34, name='dm34')
        dm45 = tf.identity(dm45, name='dm45')
        dm56 = tf.identity(dm56, name='dm56')
        qi = tf.identity(qi, name='qi')
        rain = tf.identity(rain, name='rain')
        et0 = tf.identity(et0, name='et0')
        t = tf.identity(t, name='t')



        # Outflow layer 1
        with tf.GradientTape() as tape:
            tape.watch(m1)
            x1 = self.f1_dense1(m1)
            x1 = self.f1_dense2(x1)
            x1 = self.f1_dense3(x1)
            f1 = self.f1_out(x1)
        df1_dm1 = tape.gradient(f1, m1)

        # Outflow layer 2
        with tf.GradientTape() as tape:
            tape.watch(m2)
            x2 = self.f2_dense1(m2)
            x2 = self.f2_dense2(x2)
            x2 = self.f2_dense3(x2)
            f2 = self.f2_out(x2)
        df2_dm2 = tape.gradient(f2, m2)

        # Outflow layer 3
        with tf.GradientTape() as tape:
            tape.watch(m3)
            x3 = self.f3_dense1(m3)
            x3 = self.f3_dense2(x3)
            x3 = self.f3_dense3(x3)
            f3 = self.f3_out(x3)
        df3_dm3 = tape.gradient(f3, m3)

        # Outflow layer 4
        with tf.GradientTape() as tape:
            tape.watch(m4)
            x4 = self.f4_dense1(m4)
            x4 = self.f4_dense2(x4)
            x4 = self.f4_dense3(x4)
            f4 = self.f4_out(x4)
        df4_dm4 = tape.gradient(f4, m4)

        # Outflow layer 5
        with tf.GradientTape() as tape:
            tape.watch(m5)
            x5 = self.f5_dense1(m5)
            x5 = self.f5_dense2(x5)
            x5 = self.f5_dense3(x5)
            f5 = self.f5_out(x5)
        df5_dm5 = tape.gradient(f5, m5)

        # Outflow layer 6
        with tf.GradientTape() as tape:
            tape.watch(m6)
            x6 = self.f6_dense1(m6)
            x6 = self.f6_dense2(x6)
            x6 = self.f6_dense3(x6)
            f6 = self.f6_out(x6)
        df6_dm6 = tape.gradient(f6, m6)

        # Interlayer f12
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(m1)
            tape.watch(m2)
            m12_input = tf.concat([m1, m2], axis=1)
            x = self.f12_dense1(m12_input)
            x = self.f12_dense2(x)
            x = self.f12_dense3(x)
            f12 = self.f12_out(x)
        df12_dm1 = tape.gradient(f12, m1)
        df12_dm2 = tape.gradient(f12, m2)
        df12_ddm12 = df12_dm1 - df12_dm2

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(m2)
            tape.watch(m3)
            m23_input = tf.concat([m2, m3], axis=1)
            x = self.f23_dense1(m23_input)
            x = self.f23_dense2(x)
            x = self.f23_dense3(x)
            f23 = self.f23_out(x)
        df23_dm2 = tape.gradient(f23, m2)
        df23_dm3 = tape.gradient(f23, m3)
        df23_ddm23 = df23_dm2 - df23_dm3

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(m3)
            tape.watch(m4)
            m34_input = tf.concat([m3, m4], axis=1)
            x = self.f34_dense1(m34_input)
            x = self.f34_dense2(x)
            x = self.f34_dense3(x)
            f34 = self.f34_out(x)
        df34_dm3 = tape.gradient(f34, m3)
        df34_dm4 = tape.gradient(f34, m4)
        df34_ddm34 = df34_dm3 - df34_dm4

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(m4)
            tape.watch(m5)
            m45_input = tf.concat([m4, m5], axis=1)
            x = self.f45_dense1(m45_input)
            x = self.f45_dense2(x)
            x = self.f45_dense3(x)
            f45 = self.f45_out(x)
        df45_dm4 = tape.gradient(f45, m4)
        df45_dm5 = tape.gradient(f45, m5)
        df45_ddm45 = df45_dm4 - df45_dm5

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(m5)
            tape.watch(m6)
            m56_input = tf.concat([m5, m6], axis=1)
            x = self.f56_dense1(m56_input)
            x = self.f56_dense2(x)
            x = self.f56_dense3(x)
            f56 = self.f56_out(x)
        df56_dm5 = tape.gradient(f56, m5)
        df56_dm6 = tape.gradient(f56, m6)
        df56_ddm56 = df56_dm5 - df56_dm6

        # Inflow
        with tf.GradientTape() as tape:
            tape.watch(qi)
            x = self.inflow_dense1(qi)
            x = self.inflow_dense2(x)
            x = self.inflow_dense3(x)
            inflow = self.inflow_out(x)
        dinflow_dq = tape.gradient(inflow, qi)

        # Rain
        with tf.GradientTape() as tape:
            tape.watch(rain)
            x = self.rain1_dense1(rain)
            x = self.rain1_dense2(x)
            x = self.rain1_dense3(x)
            in_rain1 = self.rain1_out(x)
        din_rain1_dq = tape.gradient(in_rain1, rain)

        with tf.GradientTape() as tape:
            tape.watch(rain)
            x = self.rain2_dense1(rain)
            x = self.rain2_dense2(x)
            x = self.rain2_dense3(x)
            in_rain2 = self.rain2_out(x)
        din_rain2_dq = tape.gradient(in_rain2, rain)

        with tf.GradientTape() as tape:
            tape.watch(rain)
            x = self.rain3_dense1(rain)
            x = self.rain3_dense2(x)
            x = self.rain3_dense3(x)
            in_rain3 = self.rain3_out(x)
        din_rain3_dq = tape.gradient(in_rain3, rain)

        with tf.GradientTape() as tape:
            tape.watch(rain)
            x = self.rain4_dense1(rain)
            x = self.rain4_dense2(x)
            x = self.rain4_dense3(x)
            in_rain4 = self.rain4_out(x)
        din_rain4_dq = tape.gradient(in_rain4, rain)

        with tf.GradientTape() as tape:
            tape.watch(rain)
            x = self.rain5_dense1(rain)
            x = self.rain5_dense2(x)
            x = self.rain5_dense3(x)
            in_rain5 = self.rain5_out(x)
        din_rain5_dq = tape.gradient(in_rain5, rain)

        with tf.GradientTape() as tape:
            tape.watch(rain)
            x = self.rain6_dense1(rain)
            x = self.rain6_dense2(x)
            x = self.rain6_dense3(x)
            in_rain6 = self.rain6_out(x)
        din_rain6_dq = tape.gradient(in_rain6, rain)




        # Plant coef 1
                # Plant coef 1
        inputs_pc1 = tf.concat([m1, et0], axis=1)
        with tf.GradientTape() as tape:
            tape.watch(inputs_pc1)
            x_pc1 = self.pc1_dense1(inputs_pc1)
            x_pc1 = self.pc1_dense2(x_pc1)
            x_pc1 = self.pc1_dense3(x_pc1)
            x_pc1 = self.pc1_dense4(x_pc1)
            pc1 = self.pc1_out(x_pc1)
        dpc1_dinputs = tape.gradient(pc1, inputs_pc1)
        dpc1_dm1 = dpc1_dinputs[:, 0:1]
        dpc1_det0 = dpc1_dinputs[:, 1:2]

        # Plant coef 2
        inputs_pc2 = tf.concat([m2, et0], axis=1)
        with tf.GradientTape() as tape:
            tape.watch(inputs_pc2)
            x_pc2 = self.pc2_dense1(inputs_pc2)
            x_pc2 = self.pc2_dense2(x_pc2)
            x_pc2 = self.pc2_dense3(x_pc2)
            x_pc2 = self.pc2_dense4(x_pc2)
            pc2 = self.pc2_out(x_pc2)
        dpc2_dinputs = tape.gradient(pc2, inputs_pc2)
        dpc2_dm2 = dpc2_dinputs[:, 0:1]
        dpc2_det0 = dpc2_dinputs[:, 1:2]

        # Plant coef 3
        inputs_pc3 = tf.concat([m3, et0], axis=1)
        with tf.GradientTape() as tape:
            tape.watch(inputs_pc3)
            x_pc3 = self.pc3_dense1(inputs_pc3)
            x_pc3 = self.pc3_dense2(x_pc3)
            x_pc3 = self.pc3_dense3(x_pc3)
            x_pc3 = self.pc3_dense4(x_pc3)
            pc3 = self.pc3_out(x_pc3)
        dpc3_dinputs = tape.gradient(pc3, inputs_pc3)
        dpc3_dm3 = dpc3_dinputs[:, 0:1]
        dpc3_det0 = dpc3_dinputs[:, 1:2]

        # Plant coef 4
        inputs_pc4 = tf.concat([m4, et0], axis=1)
        with tf.GradientTape() as tape:
            tape.watch(inputs_pc4)
            x_pc4 = self.pc4_dense1(inputs_pc4)
            x_pc4 = self.pc4_dense2(x_pc4)
            x_pc4 = self.pc4_dense3(x_pc4)
            x_pc4 = self.pc4_dense4(x_pc4)
            pc4 = self.pc4_out(x_pc4)
        dpc4_dinputs = tape.gradient(pc4, inputs_pc4)
        dpc4_dm4 = dpc4_dinputs[:, 0:1]
        dpc4_det0 = dpc4_dinputs[:, 1:2]

        # Plant coef 5
        inputs_pc5 = tf.concat([m5, et0], axis=1)
        with tf.GradientTape() as tape:
            tape.watch(inputs_pc5)
            x_pc5 = self.pc5_dense1(inputs_pc5)
            x_pc5 = self.pc5_dense2(x_pc5)
            x_pc5 = self.pc5_dense3(x_pc5)
            x_pc5 = self.pc5_dense4(x_pc5)
            pc5 = self.pc5_out(x_pc5)
        dpc5_dinputs = tape.gradient(pc5, inputs_pc5)
        dpc5_dm5 = dpc5_dinputs[:, 0:1]
        dpc5_det0 = dpc5_dinputs[:, 1:2]

        # Plant coef 6
        inputs_pc6 = tf.concat([m6, et0], axis=1)
        with tf.GradientTape() as tape:
            tape.watch(inputs_pc6)
            x_pc6 = self.pc6_dense1(inputs_pc6)
            x_pc6 = self.pc6_dense2(x_pc6)
            x_pc6 = self.pc6_dense3(x_pc6)
            x_pc6 = self.pc6_dense4(x_pc6)
            pc6 = self.pc6_out(x_pc6)
        dpc6_dinputs = tape.gradient(pc6, inputs_pc6)
        dpc6_dm6 = dpc6_dinputs[:, 0:1]
        dpc6_det0 = dpc6_dinputs[:, 1:2]



        # Lateral outflows
        with tf.GradientTape() as tape:
            tape.watch(m1)
            x_fl1 = self.fl1_dense1(m1)
            x_fl1 = self.fl1_dense2(x_fl1)
            x_fl1 = self.fl1_dense3(x_fl1)
            fl1 = self.fl1_out(x_fl1)
        dfl1_dm1 = tape.gradient(fl1, m1)

        with tf.GradientTape() as tape:
            tape.watch(m2)
            x_fl2 = self.fl2_dense1(m2)
            x_fl2 = self.fl2_dense2(x_fl2)
            x_fl2 = self.fl2_dense3(x_fl2)
            fl2 = self.fl2_out(x_fl2)
        dfl2_dm2 = tape.gradient(fl2, m2)

        with tf.GradientTape() as tape:
            tape.watch(m3)
            x_fl3 = self.fl3_dense1(m3)
            x_fl3 = self.fl3_dense2(x_fl3)
            x_fl3 = self.fl3_dense3(x_fl3)
            fl3 = self.fl3_out(x_fl3)
        dfl3_dm3 = tape.gradient(fl3, m3)

        with tf.GradientTape() as tape:
            tape.watch(m4)
            x_fl4 = self.fl4_dense1(m4)
            x_fl4 = self.fl4_dense2(x_fl4)
            x_fl4 = self.fl4_dense3(x_fl4)
            fl4 = self.fl4_out(x_fl4)
        dfl4_dm4 = tape.gradient(fl4, m4)

        with tf.GradientTape() as tape:
            tape.watch(m5)
            x_fl5 = self.fl5_dense1(m5)
            x_fl5 = self.fl5_dense2(x_fl5)
            x_fl5 = self.fl5_dense3(x_fl5)
            fl5 = self.fl5_out(x_fl5)
        dfl5_dm5 = tape.gradient(fl5, m5)

        

        return (
            f1, f2, f3, f4, f5, f6,
            f12, f23, f34, f45, f56,
            inflow,
            in_rain1, in_rain2, in_rain3, in_rain4, in_rain5, in_rain6,
            pc1, pc2, pc3, pc4, pc5, pc6,
            fl1, fl2, fl3, fl4, fl5,
            df1_dm1, df2_dm2, df3_dm3, df4_dm4, df5_dm5, df6_dm6,
            df12_ddm12, df23_ddm23, df34_ddm34, df45_ddm45, df56_ddm56,
            dinflow_dq,
            din_rain1_dq, din_rain2_dq, din_rain3_dq, din_rain4_dq, din_rain5_dq, din_rain6_dq,
            dpc1_dm1, dpc2_dm2, dpc3_dm3, dpc4_dm4, dpc5_dm5, dpc6_dm6,
            dpc1_det0, dpc2_det0, dpc3_det0, dpc4_det0, dpc5_det0, dpc6_det0,
            dfl1_dm1, dfl2_dm2, dfl3_dm3, dfl4_dm4, dfl5_dm5
        )

# Create an instance of the PINN model
