import pandas as pd
import streamlit as st
import joblib
import os

# 读取‘化学成分’工作簿
def hxcf(file_path):
    if not os.path.exists(file_path):
        st.error(f"文件 '{file_path}' 未找到，请检查文件路径。")
        st.stop()
    try:
        chengfen_df = pd.read_excel(file_path, sheet_name='化学成分')
        return chengfen_df
    except Exception as e:
        st.error(f"读取文件时发生错误: {e}")
        st.stop()

# 读取‘物料配比’工作簿
def wlpb(file_path):
    if not os.path.exists(file_path):
        st.error(f"文件 '{file_path}' 未找到，请检查文件路径。")
        st.stop()
    try:
        peibi_df = pd.read_excel(file_path, sheet_name='物料配比')
        return peibi_df
    except Exception as e:
        st.error(f"读取文件时发生错误: {e}")
        st.stop()

# 保存数据到 Excel 文件
def save_data(df, file_path, sheet_name):
    try:
        with pd.ExcelWriter(file_path, mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        st.success("数据已成功保存！")
    except Exception as e:
        st.error(f"保存数据时发生错误: {e}")
# 计算特征值
def jscf_sum(chengfen, peibi, mix_water, layer_thickness):
    # 清空相关数据结构
    result_df = pd.DataFrame()

    # 获取成分的"物料"的内容
    wuliao_name_sum = chengfen['物料'].tolist()

    # 创建一个新的 DataFrame 来存储结果，包含“残存”列
    result_columns = chengfen.columns.drop('物料').tolist() + ['残存']
    result_df = pd.DataFrame(columns=result_columns)

    # 遍历配比每一行
    for index, row in peibi.iterrows():
        # 初始化一个空的字典来存储当前行的结果，包含“残存”列
        current_result = {col: 0 for col in result_columns}

        # 遍历当前行的每个元素的值及其列名
        for column_name, value in row.items():
            if column_name in wuliao_name_sum and not pd.isna(value):  # 如果当前列名在物料名列表中且配比值不为 NaN
                # 找到成分表中对应的行
                matching_row = chengfen[chengfen['物料'] == column_name]

                if not matching_row.empty:  # 确保匹配行不为空
                    # 计算干基配比
                    h2o_value = matching_row['H2O'].values[0]   # 将水分含量转换为小数
                    dry_base_ratio = value * (100 - h2o_value) / 100

                    # 逐个乘以干基配比中的值
                    for col in matching_row.columns:
                        if col != '物料':
                            result = matching_row[col].values[0] * dry_base_ratio
                            current_result[col] += result

                    # 计算残存
                    shaosun_value = matching_row['烧损'].values[0]   # 将烧损值转换为小数
                    residue = (100 - shaosun_value) * dry_base_ratio / 100
                    current_result['残存'] += residue  # 累加残存值

        # 将当前行的结果添加到 result_df
        result_df.loc[index] = current_result

    # 计算最终结果
    for index, row in result_df.iterrows():
        # 获取当前行的残存值
        residue = row['残存']

        # 遍历当前行的每个元素，除以残存值
        for col in result_df.columns:
            if col != '残存':
                result_df.at[index, col] /= residue

    # 删除“残存”和“烧损”列
    result_df = result_df.drop(columns=['残存', '烧损'])

    # 将混合料水分和料层厚度追加到特征值计算后的最后两列
    result_df['混合料水分'] = mix_water
    result_df['料层厚度'] = layer_thickness

    return result_df

# 设置页面标题
st.title('烧结矿质量预测')

# 读取化学成分数据
file_path = '回归特征值计算.xlsx'
chengfen_df = hxcf(file_path)

# 读取物料配比数据
peibi_df = wlpb(file_path)

# 创建导航栏
page = st.sidebar.selectbox("选择页面", ["预测页面", "铁粉成分"])

if page == "预测页面":
    # 创建一个空的占位符，用于显示特征值计算结果
    result_placeholder = st.empty()

    # 创建用户输入界面
    st.write("请输入各种铁精粉的配比数据进行预测")

    # 使用表单
    with st.form("input_form"):
        # 创建5列布局
        cols = st.columns(5)
        input_ratios = []
        for i in range(len(chengfen_df)):
            val = cols[i % 5].number_input(chengfen_df['物料'].iloc[i], min_value=0.0, max_value=100.0, value=0.0, step=1.0)
            input_ratios.append(val)

        # 输入混合料水分和料层厚度
        mix_water = st.number_input("混合料水分", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        layer_thickness = st.number_input("料层厚度", min_value=0.0, max_value=1000.0, value=0.0, step=1.0)

        # 提交按钮
        submitted = st.form_submit_button("预测")

    # 如果用户提交了表单
    if submitted:

        # 清空之前的特征值计算结果显示
        result_placeholder.empty()

        # 清空相关数据结构
        result_df = pd.DataFrame()
        temp_peibi_df = pd.DataFrame()

        # 创建一个临时的配比 DataFrame
        temp_peibi_df = pd.DataFrame([input_ratios], columns=chengfen_df['物料'])

        # 计算特征值
        result_df = jscf_sum(chengfen_df, temp_peibi_df, mix_water, layer_thickness)

        # 将结果数据框中的所有数值列保留两位小数
        result_df = result_df.round(3)

        # 显示特征值计算结果
        st.write("特征值计算结果：")
        st.dataframe(result_df)

        # 加载标准化器
        try:
            scaler = joblib.load('x_scaler.pkl')
        except FileNotFoundError:
            st.error("文件 'x_scaler.pkl' 未找到，请检查文件路径。")
            st.stop()

        # 标准化特征
        X_new = scaler.transform(result_df)

        # # 加载PCA模型
        # try:
        #     pca = joblib.load('pca_model.pkl')
        # except FileNotFoundError:
        #     st.error("文件 'pca_model.pkl' 未找到，请检查文件路径。")
        #     st.stop()
        #
        # # 应用PCA降维
        # X_new = pca.transform(X_new)

        # 加载最优模型
        try:
            best_model = joblib.load('1026best_xgboost_model.pkl')
        except FileNotFoundError:
            st.error("文件 'best_xgboost_model.pkl' 未找到，请检查文件路径。")
            st.stop()

        # 预测
        y_pred = best_model.predict(X_new)

        # 显示预测结果
        st.write(f"预测垂烧速度: {y_pred[0][0]:.3f}")
        st.write(f"预测成品率: {y_pred[0][1]:.3f}")
        st.write(f"预测转鼓强度: {y_pred[0][2]:.3f}")
        st.write(f"预测平均粒级: {y_pred[0][3]:.3f}")

if page == "铁粉成分":
    st.write("以下是铁粉成分的详细信息：")
    st.dataframe(chengfen_df)

    st.subheader('编辑铁粉成分')
    # 输入密码
    password = st.text_input("请输入密码", type="password")
    if password == "123456":  # 替换为实际的密码
        edited_chengfen_df = st.session_state.get('edited_chengfen_df', chengfen_df.copy())

        # 显示数据编辑器
        edited_chengfen_df = st.data_editor(edited_chengfen_df, num_rows="dynamic")

        # 保存更新后的数据到 session_state
        st.session_state.edited_chengfen_df = edited_chengfen_df

        # 选择物料并在其上方插入新行
        material_to_insert = st.selectbox('选择要在其上方插入新行的物料', options=edited_chengfen_df['物料'].tolist())
        if st.button('在选中物料上方插入新行'):
            index_to_insert = edited_chengfen_df[edited_chengfen_df['物料'] == material_to_insert].index[0]
            new_row = {col: None for col in edited_chengfen_df.columns}
            edited_chengfen_df = pd.concat([edited_chengfen_df.iloc[:index_to_insert],
                                            pd.DataFrame([new_row]),
                                            edited_chengfen_df.iloc[index_to_insert:]],
                                           ignore_index=True)
            # 更新 session_state
            st.session_state.edited_chengfen_df = edited_chengfen_df
            st.data_editor(edited_chengfen_df, num_rows="dynamic")  # 重新显示更新后的数据框

        # 删除选中的行
        selected_names = st.multiselect('选择要删除的铁粉名称', options=edited_chengfen_df['物料'])
        if st.button('删除选中的行'):
            if selected_names:
                edited_chengfen_df = edited_chengfen_df[~edited_chengfen_df['物料'].isin(selected_names)].reset_index(
                    drop=True)
                # 更新 session_state
                st.session_state.edited_chengfen_df = edited_chengfen_df
                st.data_editor(edited_chengfen_df, num_rows="dynamic")  # 重新显示更新后的数据框
            else:
                st.warning("请选择要删除的铁粉名称。")

        # 保存编辑后的数据
        if st.button('保存修改'):
            save_data(edited_chengfen_df, file_path, sheet_name='化学成分')
    else:
        st.warning("密码错误，请重新输入。")
