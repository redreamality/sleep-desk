# 0.4.1 轻量频谱试用版

这是可回听、可人工纠正的试用版，不是经过独立人群验证的声音识别或医学模型。
它保留已知误报/漏检风险，不将开发集分数作为上线准确率。

## 当前流程

采集与声音检测、2秒前后文、后台队列、AAC保存、原事件ID/revision和人工复核不变。
后台分类改用 `SpectralSnoreClassifier`，对完整播放片段的16kHz PCM计算48维
包络、频谱形状和频谱变化特征，再使用冻结线性权重。
支持度是未经校准的分数变换，不是正确概率。阈值仍为冻结的线性决策边界0，
本次移植没有根据新的人工标签重新训练或调阈值。

- 阳性：`SNORE / SUGGESTED`，展示为模型建议。
- 阴性：`UNKNOWN / UNCERTAIN`，不伪造“静音”“说话”或“翻身”结论。
- 队列取消/模型错误：继续明确记录失败，不能删除已检测事件。
- 当前按整段音频判断，声音可能来自前后文，不能当作精确本体定位。
- 既有 `userLabel`、复核标记和旧夜晚原始分类保持不变。
- `seg-v4` 将未分类与已识别声音分开合段，旧分段索引按版本重建。

## 人声隐私

轻量二分类器不能证明录音中没有人声，`contextSpeechScore` 始终不可用。
当 `saveSpeechClips=false` 时，无法排除人声的片段全部禁止写音频，包括鼾声阳性。
事件与分类结果仍保留。不要为增加可播放片段而把缺失的人声证据当作0。

## 打包

主应用不依赖 TensorFlow Lite，也不加载仅存放历史模型的 `src/main/assets` 目录。
历史 YAMNet 类、运行时和模型资产仅放入 instrumentation 测试APK，
用于保留旧模型的回归对照，不进入交付给用户的应用APK。
`verify_spectral_apk.py` 拒绝携带 `.tflite`、TensorFlow运行时或录音文件的应用包。

主程序参数仅含特征名、权重与偏置，不带训练录音、人工标签、文件路径或来源身份。
原始私有回放夹具不会提交，也不打进任一APK。

## 一致性与测试

Python规范仍为 `tools/audio_feature_study.py` 中的 env/spec/dyn 特征子集。
Kotlin实现包括双精度FFT、短时谱统计、Butterworth/Hilbert/多相重采样和包络调制。
使用 `export_spectral_parameters.py` 导出纯数值参数；
`generate_spectral_fixtures.py` 生成可公开的纯合成参考值。

- `testDebugUnitTest`：合成信号、DSP、48特征、冻结分数、隐私和事件回写测试。
- `SpectralNativeParityTest`：Android上重算合成参考值，检查主应用没有大模型资产。
- `FullAudioPipelineTest`：实际轻量分类、AAC、存储与导出链路。
- 可选 `SLEEP_DESK_SPECTRAL_PARITY`：本机私有PCM16逐特征对照，不在CI中使用私人音频。
- `tools/verify-android-isolated.ps1 -SourceStable`：只在专用模拟器执行完整验收，
  不能在保存个人记录的手机上运行会清空测试状态的仪器测试。

数值一致不等于分类准确；合成测试和模拟器通过也不等于整夜真机功耗验证。
对等高包络峰的竞争排序仍可能与NumPy实现存在边缘差别，应持续用构造样本检查。
