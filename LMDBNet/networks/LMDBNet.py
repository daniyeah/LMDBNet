from thop import profile
from timm.models.layers import trunc_normal_
from networks.submodule import *
# from submodule import *

class LMDBNet(nn.Module):

    def __init__(self,):
        super().__init__()
        c_list = [64, 128, 256, 512] # 1 global_ch 32
        c_list1 = [32, 64, 128, 256] # 2 global_ch 32
        c_list2 = [16, 32, 64, 128]  # 3 global_ch 32
        c_list3 = [8, 16, 32, 64]    # 4 global_ch 16
        c_list4 = [4, 8, 16, 32]     # 5 global_ch 8

        self.encoder1 = DoubleConv(3, c_list[0])
        self.encoder2 = DoubleConv(c_list[0], c_list[1])
        self.encoder3 = DoubleConv(c_list[1], c_list[2])
        self.encoder4 = DoubleConv(c_list[2], c_list[3])

        self.encoder12 = DoubleConv(3, c_list[0])
        self.encoder22 = DoubleConv(c_list[0], c_list[1])
        self.encoder32 = DoubleConv(c_list[1], c_list[2])
        self.encoder42 = DoubleConv(c_list[2], c_list[3])

        self.cross_cat4 = Cross_cat(c_list[3])
        self.cross_cat3 = Cross_cat(c_list[2])
        self.cross_cat2 = Cross_cat(c_list[1])
        self.cross_cat1 = Cross_cat(c_list[0])

        self.decoder0 = Decoder(c_list)
        self.decoder1 = Decoder(c_list1)
        self.decoder2 = Decoder(c_list2)
        self.decoder3 = Decoder(c_list3)
        self.decoder4 = Decoder(c_list4)

        self.glo_feature = Global_feature(32, c_list)   # 1
        self.glo_feature1 = Global_feature(32, c_list1) # 2
        self.glo_feature2 = Global_feature(32, c_list2) # 3
        self.glo_feature3 = Global_feature(16, c_list3) # 4
        self.glo_feature4 = Global_feature(8, c_list4)  # 5

        self.pool = nn.MaxPool2d(2, 2)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        out11 = self.encoder1(x1)
        out12 = self.encoder12(x2)
        t1 = self.cross_cat1(out11, out12)  # 256 256
        out11 = self.pool(out11)
        out12 = self.pool(out12)

        out21 = self.encoder2(out11)
        out22 = self.encoder22(out12)
        t2 = self.cross_cat2(out21, out22)  # 128 128
        out21 = self.pool(out21)
        out22 = self.pool(out22)

        out31 = self.encoder3(out21)
        out32 = self.encoder32(out22)
        t3 = self.cross_cat3(out31, out32)  # 64 64
        out31 = self.pool(out31)
        out32 = self.pool(out32)

        out41 = self.encoder4(out31)
        out42 = self.encoder42(out32)
        t4 = self.cross_cat4(out41, out42)  # 32 32

        # 1
        glo_deep, global_out1, global_out2, global_out3, global_out4 = self.glo_feature(t1, t2, t3, t4)
        out, out1, out2, out3, out4 = self.decoder0(global_out1, global_out2, global_out3, global_out4)
        # 2
        glo_deep1, g_de1_out1, g_de1_out2, g_de1_out3, g_de1_out4 = self.glo_feature1(out1, out2, out3, out4)
        decoder_out1, de1_out1, de1_out2, de1_out3, de1_out4 = self.decoder1(g_de1_out1, g_de1_out2, g_de1_out3, g_de1_out4)
        # 3
        glo_deep2, g_de2_out1, g_de2_out2, g_de2_out3, g_de2_out4 = self.glo_feature2(de1_out1, de1_out2, de1_out3, de1_out4)
        decoder_out2, de2_out1, de2_out2, de2_out3, de2_out4 = self.decoder2(g_de2_out1, g_de2_out2, g_de2_out3, g_de2_out4)
        # 4
        # glo_deep3, g_de3_out1, g_de3_out2, g_de3_out3, g_de3_out4 = self.glo_feature3(de2_out1, de2_out2, de2_out3, de2_out4)
        # decoder_out3, de3_out1, de3_out2, de3_out3, de3_out4 = self.decoder3(g_de3_out1, g_de3_out2, g_de3_out3, g_de3_out4)

        # 5
        # glo_deep4, g_de4_out1, g_de4_out2, g_de4_out3, g_de4_out4 = self.glo_feature4(de3_out1, de3_out2, de3_out3, de3_out4)
        # decoder_out4, de4_out1, de4_out2, de4_out3, de4_out4 = self.decoder4(g_de4_out1, g_de4_out2, g_de4_out3, g_de4_out4)



        # return (out, glo_deep) # 1
        # return (decoder_out1, out, glo_deep, glo_deep1) # 2
        return (decoder_out2, decoder_out1, out, glo_deep, glo_deep1, glo_deep2) # 3
        # return (decoder_out3, decoder_out2, decoder_out1, out, glo_deep, glo_deep1, glo_deep2, glo_deep3)  # 4
        # return (decoder_out4,decoder_out3, decoder_out2, decoder_out1, out, glo_deep, glo_deep1, glo_deep2, glo_deep3,glo_deep4)  # 5
