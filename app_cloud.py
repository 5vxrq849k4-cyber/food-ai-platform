"""
食品与AI联合分析平台 - 云端部署版本 v2.1
Food & AI Analysis Platform - Cloud Deployment Version

适用于 Streamlit Cloud 部署
访问方式: https://your-app-name.streamlit.app

作者: AI Assistant
日期: 2026-04-29
更新: v2.1 - 修复依赖问题
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc
)
import warnings
import time

warnings.filterwarnings('ignore')

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="食品与AI联合分析平台",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 配色方案
# ============================================================
COLORS = {
    'primary': '#1f4e79',
    'secondary': '#2e75b6',
    'accent': '#5b9bd5',
    'success': '#70ad47',
    'warning': '#ffc000',
    'danger': '#c00000',
    'neutral': '#7f7f7f',
}

# ============================================================
# 预处理函数
# ============================================================
def snv(spectra):
    """标准正态变量变换"""
    mean = np.mean(spectra, axis=1, keepdims=True)
    std = np.std(spectra, axis=1, keepdims=True)
    std[std == 0] = 1
    return (spectra - mean) / std

def preprocess_pipeline(X, snv_enable=True, sg_smooth=True,
                        sg_window=11, normalize=True):
    """预处理流程"""
    X = X.copy().astype(float)

    if snv_enable:
        X = snv(X)

    if sg_smooth:
        X = savgol_filter(X, sg_window, 2, deriv=0, axis=1)

    if normalize:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    return np.nan_to_num(X, nan=0.0)

# ============================================================
# 可视化函数
# ============================================================
def plot_spectra_plotly(X, wavelengths, title, sample_indices=None,
                        categories=None, height=350):
    """绘制光谱图"""
    fig = go.Figure()

    if sample_indices is None:
        sample_indices = list(range(min(5, len(X))))

    colors = px.colors.qualitative.Set1

    for i, idx in enumerate(sample_indices[:5]):
        label = f"Sample {idx}"
        if categories is not None and idx < len(categories):
            label += f" ({categories[idx]})"

        fig.add_trace(go.Scatter(
            x=wavelengths,
            y=X[idx],
            mode='lines',
            name=label,
            line=dict(color=colors[i % len(colors)], width=1)
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Wavenumber (cm⁻¹)' if len(wavelengths) > 5000 else 'Raman Shift (cm⁻¹)',
        yaxis_title='Intensity',
        height=height,
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(font=dict(size=9))
    )

    return fig

def plot_confusion_matrix_plotly(y_true, y_pred, classes, title):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=classes,
        y=classes,
        colorscale='Blues',
        showscale=True,
        text=cm,
        texttemplate='%{text}',
        textfont=dict(size=10)
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Predicted',
        yaxis_title='True',
        height=350,
        width=400
    )

    return fig

def plot_roc_curves(y_test_bin, y_score, classes, title):
    """绘制ROC曲线"""
    n_classes = len(classes)
    fig = go.Figure()

    colors = px.colors.qualitative.Set1[:min(n_classes, 20)]

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{classes[i]} (AUC={roc_auc:.3f})',
            line=dict(color=colors[i % len(colors)], width=1.5)
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='gray', width=2, dash='dash')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=400,
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(font=dict(size=8))
    )

    return fig

# ============================================================
# 主程序
# ============================================================
def main():
    # 标题
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1f4e79, #2e75b6);
                padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color: white; margin: 0; text-align: center;'>
            🧪 食品与AI联合分析平台
        </h1>
        <p style='color: rgba(255,255,255,0.8); text-align: center; margin: 5px 0 0 0;'>
            Food & AI Analysis Platform | FTIR/Raman 光谱分类分析
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        st.header("📁 数据上传")
        uploaded_file = st.file_uploader(
            "上传光谱数据",
            type=['xlsx', 'xls', 'csv'],
            help="支持 Excel (.xlsx, .xls) 或 CSV 格式"
        )

        st.divider()
        st.header("⚙️ 预处理参数")
        snv_enable = st.checkbox("SNV 标准化", value=True)
        sg_smooth = st.checkbox("SG 平滑", value=True)
        sg_window = st.select_slider("SG 窗口", options=[5, 7, 9, 11, 13, 15], value=11)
        normalize = st.checkbox("归一化", value=True)

        st.divider()
        st.header("🤖 模型参数")
        cv_folds = st.selectbox("交叉验证折数", [5, 10], index=0)
        n_estimators = st.slider("随机森林树数", 10, 200, 100)
        svm_kernel = st.selectbox("SVM 核函数", ["rbf", "linear"])

        st.divider()
        analyze_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)

    # 主内容区
    if uploaded_file is not None:
        # 加载数据
        @st.cache_data
        def load_data(file):
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            return df

        with st.spinner("正在加载数据..."):
            df = load_data(uploaded_file)

        # 提取数据
        sample_names = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values.astype(float)
        X = np.nan_to_num(X, nan=0.0)
        categories = np.array([str(name)[:3].upper() if len(str(name)) >= 3 else 'UNK'
                              for name in sample_names])

        # 检测光谱类型
        n_features = X.shape[1]
        if 7000 <= n_features <= 7200:
            spectra_type = "FTIR"
            wavelength_range = (550, 4000)
        elif 2900 <= n_features <= 3100:
            spectra_type = "Raman"
            wavelength_range = (200, 3200)
        else:
            spectra_type = "Unknown"
            wavelength_range = (0, n_features)

        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_features)

        # 数据概览
        st.subheader("📊 数据概览")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("样本数量", X.shape[0])
        with col2:
            st.metric("特征维度", X.shape[1])
        with col3:
            st.metric("类别数量", len(np.unique(categories)))
        with col4:
            st.metric("光谱类型", spectra_type)

        # 类别分布
        unique, counts = np.unique(categories, return_counts=True)
        fig_dist = go.Figure([go.Bar(x=unique, y=counts, marker_color=COLORS['primary'])])
        fig_dist.update_layout(title="类别分布", height=300)
        st.plotly_chart(fig_dist, use_container_width=True)

        # 分析按钮处理
        if analyze_btn:
            with st.spinner("正在分析..."):
                # 预处理
                X_processed = preprocess_pipeline(X, snv_enable, sg_smooth,
                                                  sg_window, normalize)

                # 编码标签
                le = LabelEncoder()
                y = le.fit_transform(categories)
                classes = le.classes_

                # 定义模型
                models = {
                    'Random Forest': RandomForestClassifier(
                        n_estimators=n_estimators, random_state=42, n_jobs=-1),
                    'SVM': SVC(kernel=svm_kernel, probability=True, random_state=42),
                    'MLP': MLPClassifier(hidden_layer_sizes=(100, 50),
                                        max_iter=500, random_state=42)
                }

                # 交叉验证
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                results = {}

                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, (name, model) in enumerate(models.items()):
                    status_text.text(f"训练 {name}...")
                    model.fit(X_processed, y)
                    y_pred = cross_val_predict(model, X_processed, y, cv=cv)
                    y_pred_proba = cross_val_predict(model, X_processed, y, cv=cv,
                                                     method='predict_proba')

                    results[name] = {
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba,
                        'accuracy': accuracy_score(y, y_pred),
                        'precision': precision_score(y, y_pred, average='weighted'),
                        'recall': recall_score(y, y_pred, average='weighted'),
                        'f1': f1_score(y, y_pred, average='weighted')
                    }

                    # 计算AUC
                    y_bin = np.eye(len(classes))[y]
                    results[name]['auc'] = 0
                    for j in range(len(classes)):
                        fpr, tpr, _ = roc_curve(y_bin[:, j], y_pred_proba[:, j])
                        results[name]['auc'] += auc(fpr, tpr)
                    results[name]['auc'] /= len(classes)

                    progress_bar.progress((i + 1) / len(models))

                progress_bar.empty()
                status_text.empty()

                # 显示结果
                st.success("分析完成！")

                # 标签页
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📈 光谱分析", "🤖 模型性能", "📉 ROC曲线", "💡 智能建议"
                ])

                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(
                            plot_spectra_plotly(X, wavelengths, "原始光谱",
                                              list(range(min(5, len(X)))), categories),
                            use_container_width=True
                        )
                    with col2:
                        st.plotly_chart(
                            plot_spectra_plotly(X_processed, wavelengths, "预处理后光谱",
                                              list(range(min(5, len(X_processed)))), categories),
                            use_container_width=True
                        )

                with tab2:
                    # 模型对比表
                    metrics_df = pd.DataFrame({
                        name: {
                            'Accuracy': f"{r['accuracy']*100:.2f}%",
                            'Precision': f"{r['precision']*100:.2f}%",
                            'Recall': f"{r['recall']*100:.2f}%",
                            'F1-Score': f"{r['f1']*100:.2f}%",
                            'AUC': f"{r['auc']*100:.2f}%"
                        }
                        for name, r in results.items()
                    }).T
                    st.dataframe(metrics_df, use_container_width=True)

                    # 混淆矩阵
                    st.subheader("混淆矩阵")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.plotly_chart(
                            plot_confusion_matrix_plotly(y, results['Random Forest']['y_pred'],
                                                        classes, "Random Forest"),
                            use_container_width=True
                        )
                    with col2:
                        st.plotly_chart(
                            plot_confusion_matrix_plotly(y, results['SVM']['y_pred'],
                                                        classes, "SVM"),
                            use_container_width=True
                        )
                    with col3:
                        st.plotly_chart(
                            plot_confusion_matrix_plotly(y, results['MLP']['y_pred'],
                                                        classes, "MLP"),
                            use_container_width=True
                        )

                with tab3:
                    # ROC曲线
                    y_bin = np.eye(len(classes))[y]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(
                            plot_roc_curves(y_bin, results['Random Forest']['y_pred_proba'],
                                          classes, "Random Forest ROC"),
                            use_container_width=True
                        )
                    with col2:
                        st.plotly_chart(
                            plot_roc_curves(y_bin, results['SVM']['y_pred_proba'],
                                          classes, "SVM ROC"),
                            use_container_width=True
                        )

                with tab4:
                    # 智能建议
                    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
                    best_acc = best_model[1]['accuracy'] * 100

                    standards = {'FTIR': {85: '合格', 92: '良好', 98: '优秀'},
                                'Raman': {75: '合格', 85: '良好', 93: '优秀'}}
                    std = standards.get(spectra_type, standards['FTIR'])

                    if best_acc >= list(std.keys())[2]:
                        level, color = '优秀', 'green'
                    elif best_acc >= list(std.keys())[1]:
                        level, color = '良好', 'blue'
                    elif best_acc >= list(std.keys())[0]:
                        level, color = '合格', 'orange'
                    else:
                        level, color = '待改进', 'red'

                    st.markdown(f"""
                    ### 📊 分析报告

                    **最佳模型**: {best_model[0]}
                    **准确率**: {best_acc:.2f}%
                    **性能等级**: :{color}[{level}]

                    ---

                    **数据质量评估**:
                    - 类别不平衡比: {max(counts)/min(counts):.1f}:1
                    - 光谱类型: {spectra_type}

                    **技术建议**:
                    - {'关注指纹区特征，优化预处理参数' if spectra_type == 'FTIR' else '增加积分时间，使用基线校正'}
                    """)

    else:
        # 欢迎界面
        st.markdown("""
        ### 👋 欢迎使用食品与AI联合分析平台

        本平台支持 **FTIR** 和 **Raman** 光谱数据的分类分析。

        #### 使用步骤：
        1. 在左侧上传 Excel 或 CSV 数据文件
        2. 调整预处理和模型参数
        3. 点击 **开始分析** 按钮

        #### 数据格式要求：
        - 第一列：样本名称（前3字母为类别标签）
        - 其余列：光谱数据

        ---

        **平台特点**：
        - ✅ 支持 FTIR/Raman 光谱分类
        - ✅ 自动检测光谱类型
        - ✅ 多种机器学习模型对比
        - ✅ 交互式可视化图表
        - ✅ 智能分析建议
        """)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info("📊 **数据上传**\nExcel/CSV格式")
        with col2:
            st.info("⚙️ **预处理**\nSNV/SG/归一化")
        with col3:
            st.info("🤖 **机器学习**\nRF/SVM/MLP")
        with col4:
            st.info("💡 **智能建议**\n客观评价")

# ============================================================
# 云端部署信息
# ============================================================
def show_cloud_info():
    """显示云端部署信息"""
    with st.expander("🌐 云端部署信息"):
        st.markdown("""
        **当前部署地址**: 本地运行

        **部署到 Streamlit Cloud**:
        1. 将代码上传到 GitHub
        2. 访问 [share.streamlit.io](https://share.streamlit.io)
        3. 连接 GitHub 仓库
        4. 选择 `app_cloud.py` 作为主文件
        5. 点击 Deploy

        部署后将获得永久访问链接，可在任何设备上使用。
        """)

if __name__ == '__main__':
    main()
    show_cloud_info()
