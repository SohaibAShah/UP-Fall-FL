import torch, torch.nn as nn
import torch.nn.functional as F


class ModelCSVIMG(nn.Module):
    # def __init__(self, num_csv_features, img_shape1, img_shape2):
    #     super(ModelCSVIMG, self).__init__()
    #
    #     # 第一输入分支：处理CSV特征的1D卷积
    #     # self.conv1d = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3)
    #     # self.pool1d = nn.MaxPool1d(kernel_size=2)
    #     # self.batch_norm1d = nn.BatchNorm1d(10)
    #     self.fc_csv_1 = nn.Linear(num_csv_features, 2000)
    #     self.bn_csv_1 = nn.BatchNorm1d(2000)
    #     self.fc_csv_2 = nn.Linear(2000, 600)
    #     self.bn_csv_2 = nn.BatchNorm1d(600)
    #     self.dropout_csv = nn.Dropout(0.2)
    #     # self.fc_csv_3 = nn.Linear(600, 12)
    #
    #     # 第二输入分支：处理第一张图像的2D卷积
    #     self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
    #     self.pool2d_1 = nn.MaxPool2d(kernel_size=2, stride=2)
    #     self.batch_norm2d_1 = nn.BatchNorm2d(16)
    #
    #     # 第三输入分支：处理第二张图像的2D卷积
    #     self.conv2d_2 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
    #     self.pool2d_2 = nn.MaxPool2d(kernel_size=2, stride=2)
    #     self.batch_norm2d_2 = nn.BatchNorm2d(16)
    #
    #     # 计算展平后的维度
    #     # self.flattened_dim_csv = (num_csv_features - 2) // 2 * 10
    #     self.flattened_dim_csv = 600
    #     self.flattened_dim_img = 16 * (img_shape1 // 2) * (img_shape2 // 2)
    #
    #     # 全连接层
    #     self.fc1 = nn.Linear(self.flattened_dim_csv + 2 * self.flattened_dim_img, 600)
    #     self.fc2 = nn.Linear(600, 1200)
    #     self.dropout = nn.Dropout(0.2)
    #     self.fc3 = nn.Linear(1200, 12)
    #
    # def forward(self, x_csv, x_img1, x_img2):
    #     # 第一分支：CSV特征
    #     # x_csv = F.relu(self.conv1d(x_csv))
    #     # x_csv = self.pool1d(x_csv)
    #     # x_csv = self.batch_norm1d(x_csv)
    #     # x_csv = x_csv.view(x_csv.size(0), -1)  # 展平
    #
    #     x_csv = F.relu(self.bn_csv_1(self.fc_csv_1(x_csv)))
    #     x_csv = F.relu(self.bn_csv_2(self.fc_csv_2(x_csv)))
    #     x_csv = self.dropout_csv(x_csv)
    #     # x_csv = self.fc_csv_3(x_csv)
    #
    #     x_img1 = x_img1.permute(0, 3, 1, 2)
    #     # 第二分支：第一张图像
    #     x_img1 = F.relu(self.conv2d_1(x_img1))
    #     x_img1 = self.pool2d_1(x_img1)
    #     x_img1 = self.batch_norm2d_1(x_img1)
    #     x_img1 = x_img1.view(x_img1.size(0), -1)  # 展平
    #
    #     x_img2 = x_img2.permute(0, 3, 1, 2)
    #     # 第三分支：第二张图像
    #     x_img2 = F.relu(self.conv2d_2(x_img2))
    #     x_img2 = self.pool2d_2(x_img2)
    #     x_img2 = self.batch_norm2d_2(x_img2)
    #     x_img2 = x_img2.view(x_img2.size(0), -1)  # 展平
    #
    #     # 连接三个分支
    #     x = torch.cat((x_csv, x_img1, x_img2), dim=1)
    #
    #     # 全连接层
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.dropout(x)
    #     x = F.softmax(self.fc3(x), dim=1)
    #
    #     return x
    def __init__(self, num_csv_features, img_shape1, img_shape2):
        super(ModelCSVIMG, self).__init__()

        # V1======================================================================================
        # # 第一输入分支：处理CSV特征的1D卷积
        # # self.conv1d = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3)
        # # self.pool1d = nn.MaxPool1d(kernel_size=2)
        # # self.batch_norm1d = nn.BatchNorm1d(10)
        # self.fc_csv_1 = nn.Linear(num_csv_features, 2000)
        # self.bn_csv_1 = nn.BatchNorm1d(2000)
        # self.fc_csv_2 = nn.Linear(2000, 600)
        # self.bn_csv_2 = nn.BatchNorm1d(600)
        # self.dropout_csv = nn.Dropout(0.2)
        # # self.fc_csv_3 = nn.Linear(600, 12)
        #
        # # 第二输入分支：处理第一张图像的2D卷积
        # self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # self.pool2d_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.batch_norm2d_1 = nn.BatchNorm2d(16)
        #
        # # 第三输入分支：处理第二张图像的2D卷积
        # self.conv2d_2 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # self.pool2d_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.batch_norm2d_2 = nn.BatchNorm2d(16)
        #
        # # 计算展平后的维度
        # # self.flattened_dim_csv = (num_csv_features - 2) // 2 * 10
        # self.flattened_dim_csv = 600
        # self.flattened_dim_img = 16 * (img_shape1 // 2) * (img_shape2 // 2)
        #
        # # 全连接层
        # self.fc1 = nn.Linear(self.flattened_dim_csv + 2 * self.flattened_dim_img, 600)
        # self.fc2 = nn.Linear(600, 1200)
        # self.dropout = nn.Dropout(0.2)
        # self.fc3 = nn.Linear(1200, 12)

        # # v2==========================================
        # # 第一输入分支：处理CSV特征
        # self.csv_fc_1 = nn.Linear(num_csv_features, 2000)
        # self.csv_bn_1 = nn.BatchNorm1d(2000)
        # self.csv_fc_2 = nn.Linear(2000, 600)
        # self.csv_bn_2 = nn.BatchNorm1d(600)
        # self.csv_fc_3 = nn.Linear(600, 100)
        # self.csv_dropout = nn.Dropout(0.2)
        #
        # # 第二输入分支：处理第一张图像的2D卷积
        # self.img1_conv_1 = nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3, stride=1, padding=1)
        # self.img1_batch_norm = nn.BatchNorm2d(18)
        # self.img1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.img1_fc1 = nn.Linear(18 * (16) * 16, 100)
        # self.img1_dropout = nn.Dropout(0.2)
        #
        # # 第三输入分支：处理第二张图像的2D卷积
        # self.img2_conv = nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3, stride=1, padding=1)
        # self.img2_batch_norm = nn.BatchNorm2d(18)
        # self.img2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.img2_fc1 = nn.Linear(18 * (16) * 16, 100)
        # self.img2_dropout = nn.Dropout(0.2)
        #
        # # 全连接层
        # self.fc1 = nn.Linear(300, 600)
        # self.fc2 = nn.Linear(600, 1200)
        # self.dropout = nn.Dropout(0.2)
        # self.fc3 = nn.Linear(1200, 12)

        # v3==========================================
        # 第一输入分支：处理CSV特征
        self.csv_fc_1 = nn.Linear(num_csv_features, 2000)
        self.csv_bn_1 = nn.BatchNorm1d(2000)
        self.csv_fc_2 = nn.Linear(2000, 600)
        self.csv_bn_2 = nn.BatchNorm1d(600)
        self.csv_dropout = nn.Dropout(0.2)

        # 第二输入分支：处理第一张图像的2D卷积
        self.img1_conv_1 = nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.img1_batch_norm = nn.BatchNorm2d(18)
        self.img1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.img1_fc1 = nn.Linear(18 * (16) * 16, 100)
        self.img1_dropout = nn.Dropout(0.2)

        # 第三输入分支：处理第二张图像的2D卷积
        self.img2_conv = nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.img2_batch_norm = nn.BatchNorm2d(18)
        self.img2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.img2_fc1 = nn.Linear(18 * (16) * 16, 100)
        self.img2_dropout = nn.Dropout(0.2)

        # 全连接层
        self.fc1 = nn.Linear(800, 1200)
        self.dr1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(2000, 12)

    def forward(self, x_csv, x_img1, x_img2):
        x_csv = F.relu(self.csv_bn_1(self.csv_fc_1(x_csv)))
        x_csv = F.relu(self.csv_bn_2(self.csv_fc_2(x_csv)))
        x_csv = self.csv_dropout(x_csv)
        # x_csv = self.fc_csv_3(x_csv)

        x_img1 = x_img1.permute(0, 3, 1, 2)
        # 第二分支：第一张图像
        x_img1 = F.relu(self.img1_conv_1(x_img1))
        x_img1 = self.img1_batch_norm(x_img1)
        x_img1 = self.img1_pool(x_img1)
        x_img1 = x_img1.contiguous().view(x_img1.size(0), -1)
        x_img1 = F.relu(self.img1_fc1(x_img1))
        x_img1 = self.img1_dropout(x_img1)

        x_img2 = x_img2.permute(0, 3, 1, 2)
        # 第三分支：第二张图像
        x_img2 = F.relu(self.img2_conv(x_img2))
        x_img2 = self.img2_batch_norm(x_img2)
        x_img2 = self.img2_pool(x_img2)
        x_img2 = x_img2.contiguous().view(x_img2.size(0), -1)
        x_img2 = F.relu(self.img2_fc1(x_img2))
        x_img2 = self.img2_dropout(x_img2)

        # 连接三个分支
        x = torch.cat((x_csv, x_img1, x_img2), dim=1)
        residual = x
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dr1(x)
        # x += residual
        x = torch.cat((residual, x), dim=1)
        x = F.softmax(self.fc2(x), dim=1)

        return x