const express = require('express');
const cors = require('cors');
const axios = require('axios');  // 添加这行

const app = express();

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Python 模型服务的地址
const MODEL_SERVICE_URL = 'http://localhost:5001';

// 上传数据接口
app.post('/api/upload', (req, res) => {
    const data = req.body;
    console.log('收到上传请求:');
    console.log('用户:', data.user_id);
    console.log('时间:', data.timestamp);
    console.log('数据条数:', data.sensor_data?.length || 0);

    res.json({
        code: 200,
        message: '数据接收成功',
        data: {
            received_count: data.sensor_data?.length || 0
        }
    });
});

// 预测接口 - 调用 Python 模型服务
app.post('/api/predict', async (req, res) => {
    // 增加：检查 req.body 是否存在
    if (!req.body) {
        return res.status(400).json({
            code: 400,
            message: '请求体不能为空',
            data: null
        });
    }

    const data = req.body;
    console.log('收到预测请求，用户:', data.user_id || 'unknown');
    console.log('IMU数据条数:', data.imu_data?.length || 0);

    // 增加：数据量检查
    if (!data.imu_data || data.imu_data.length < 100) {
        return res.status(400).json({
            code: 400,
            message: `数据量不足，至少需要100个数据点，当前${data.imu_data?.length || 0}个`,
            data: null
        });
    }

    try {
        // 调用 Python 模型服务
        const response = await axios.post(`${MODEL_SERVICE_URL}/predict`, {
            imu_data: data.imu_data,
            member_id: data.user_id || 'unknown'
        });

        res.json(response.data);

    } catch (error) {
        console.error('调用模型服务失败:', error.message);
        if (error.response) {
            console.error('模型服务返回:', error.response.data);
        }
        res.status(500).json({
            code: 500,
            message: '模型服务调用失败',
            detail: error.message,
            data: null
        });
    }
});

// 健康检查接口
app.get('/api/health', async (req, res) => {
    try {
        const response = await axios.get(`${MODEL_SERVICE_URL}/health`);
        res.json({
            node: 'ok',
            model_service: response.data
        });
    } catch (error) {
        res.json({
            node: 'ok',
            model_service: 'unavailable'
        });
    }
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`后端服务器运行在 http://localhost:${PORT}`);
    console.log(`上传接口: http://localhost:${PORT}/api/upload`);
    console.log(`预测接口: http://localhost:${PORT}/api/predict`);
    console.log(`健康检查: http://localhost:${PORT}/api/health`);
    console.log(`模型服务地址: ${MODEL_SERVICE_URL}`);
});