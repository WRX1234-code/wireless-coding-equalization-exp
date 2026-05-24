"""
Part 1：信道编码实验

学生需要完成 Hamming(7,4) 编码、伴随式计算和单比特纠错译码。
选做内容包括卷积码编码和 Viterbi 硬判决译码。
"""

import numpy as np
from utils import (
    binary_symmetric_channel,
    calculate_ber,
    generate_bits,
    plot_ber_curve,
)

HAMMING_G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=int)

HAMMING_H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1],
], dtype=int)


def hamming74_encode(bits):
    """
    Hamming(7,4) 系统码编码。

    参数:
        bits: 一维 0/1 数组，长度必须是 4 的倍数。

    返回:
        encoded: 一维 0/1 编码比特数组，长度为输入的 7/4 倍。

    要求:
        使用课件中的生成矩阵 G，按 GF(2) 进行矩阵乘法。
    """
    bits = np.asarray(bits, dtype=int)
    if bits.ndim != 1:
        raise ValueError('bits 必须是一维数组')
    if len(bits) % 4 != 0:
        raise ValueError('Hamming(7,4) 要求输入长度为 4 的倍数')
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError('bits 只能包含 0 或 1')

    # 将 bits reshape 为 (-1, 4)，再与 HAMMING_G 相乘并对 2 取模
    blocks = bits.reshape(-1, 4)
    encoded = (blocks @ HAMMING_G) % 2
    return encoded.flatten()

    # raise NotImplementedError('请实现 Hamming(7,4) 编码')


def hamming74_syndrome(codewords):
    """
    计算 Hamming(7,4) 码字的伴随式。

    参数:
        codewords: 一维或二维 0/1 数组。若为一维，长度必须是 7 的倍数。

    返回:
        syndromes: 形状为 (N, 3) 的伴随式数组。
    """
    codewords = np.asarray(codewords, dtype=int)
    if codewords.ndim == 1:
        if len(codewords) % 7 != 0:
            raise ValueError('码字长度必须是 7 的倍数')
        codewords = codewords.reshape(-1, 7)
    if codewords.shape[1] != 7:
        raise ValueError('每个 Hamming(7,4) 码字长度必须为 7')

    # 计算 s = r H^T mod 2
    syndromes = (codewords @ HAMMING_H.T) % 2
    return syndromes
    # raise NotImplementedError('请实现伴随式计算')


def hamming74_decode(received):
    """
    Hamming(7,4) 单比特纠错译码。

    参数:
        received: 一维 0/1 接收序列，长度必须是 7 的倍数。

    返回:
        decoded_bits: 纠错后提取出的信息比特序列。

    提示:
        1. 计算每个码字的伴随式。
        2. 若伴随式非零，将其与 H 的各列比较，定位错误比特。
        3. 翻转对应错误位。
        4. 系统码的信息位为前 4 位。
    """
    received = np.asarray(received, dtype=int)
    if received.ndim != 1 or len(received) % 7 != 0:
        raise ValueError('received 必须是一维数组，长度为 7 的倍数')

    # 将 received reshape 为 (-1, 7)，复制一份避免直接修改输入
    codewords = received.reshape(-1, 7).copy()
    
    # 调用 hamming74_syndrome 计算每个码字的伴随式
    syndromes = hamming74_syndrome(codewords)
    
    # 对每个非零伴随式，与 HAMMING_H 的 7 列逐列比较，定位并翻转错误位
    for i in range(codewords.shape[0]):
        syndrome = syndromes[i]
        # 若伴随式非零，说明有错误
        if not np.all(syndrome == 0):
            # 与 HAMMING_H 的每一列比较，找到匹配列
            for col_idx in range(HAMMING_H.shape[1]):
                if np.array_equal(HAMMING_H[:, col_idx], syndrome):
                    # 翻转对应码字位置
                    codewords[i, col_idx] ^= 1
                    break
    
    # 系统码的信息位为前 4 位，取每个码字前 4 位并 flatten 返回
    decoded_bits = codewords[:, :4].flatten()
    return decoded_bits
    # raise NotImplementedError('请实现 Hamming(7,4) 译码')


def convolutional_encode(bits):
    """
    选做：实现 (2,1,3) 卷积码编码，生成多项式为 g1=111, g2=101。

    默认在末尾添加 2 个 0 作为尾比特，使状态回到全零。
    """
    bits = np.asarray(bits, dtype=int)
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError('bits 只能包含 0 或 1')

    # 生成多项式 g1=111, g2=101
    # g1 对应输出1: 当前位 + 前一位 + 前两位
    # g2 对应输出2: 当前位 + 前两位
    
    # 在末尾添加 2 个尾比特 0
    bits_padded = np.concatenate([bits, [0, 0]])
    
    encoded = []
    # 移位寄存器初始状态为 [0, 0]（存储前两位）
    state = [0, 0]
    
    for bit in bits_padded:
        # 当前输入为 bit，寄存器状态为 [state[0], state[1]] = [前一位, 前两位]
        # 输出1 = bit XOR state[0] XOR state[1] (g1=111)
        out1 = (bit ^ state[0] ^ state[1]) & 1
        # 输出2 = bit XOR state[1] (g2=101)
        out2 = (bit ^ state[1]) & 1
        
        encoded.append(out1)
        encoded.append(out2)
        
        # 更新移位寄存器：新状态 = [当前bit, 前一位]
        state = [bit, state[0]]
    
    return np.array(encoded, dtype=int)
    # raise NotImplementedError('选做：请实现卷积码编码')


def viterbi_decode_hard(received_bits):
    """
    选做：实现 (2,1,3) 卷积码硬判决 Viterbi 译码。
    """
    received_bits = np.asarray(received_bits, dtype=int)
    if len(received_bits) % 2 != 0:
        raise ValueError('卷积码接收序列长度必须是 2 的倍数')

    # (2,1,3) 卷积码，约束长度 K=3，状态数为 2^(K-1) = 4
    # 状态用 2 位表示：[s1, s0]，s1 为较新的位，s0 为较旧的位
    # 状态转移：输入 0 -> 新状态 = [0, s1]，输入 1 -> 新状态 = [1, s1]
    
    num_states = 4  # 2^(3-1) = 4
    # 将接收序列按每 2 位分组
    symbols = received_bits.reshape(-1, 2)
    num_steps = len(symbols)
    
    # 初始化路径度量，状态 0 为 0，其他为无穷大
    INF = float('inf')
    path_metrics = [INF] * num_states
    path_metrics[0] = 0
    
    # 保存每个状态在每个时刻的前驱状态和输入比特
    # trellis[state][step] = (prev_state, input_bit)
    trellis = [[None for _ in range(num_steps)] for _ in range(num_states)]
    
    # 状态转移表和输出表
    # 状态 s = (s1 << 1) | s0，即 s1 是高位，s0 是低位
    # 对于状态 [s1, s0]，输入 u 时：
    #   输出1 = u XOR s1 XOR s0
    #   输出2 = u XOR s0
    #   下一状态 = [u, s1]
    
    for step in range(num_steps):
        new_metrics = [INF] * num_states
        for state in range(num_states):
            if path_metrics[state] == INF:
                continue
            s1 = (state >> 1) & 1  # 高位
            s0 = state & 1          # 低位
            
            # 尝试输入 0 和 1
            for input_bit in [0, 1]:
                # 计算输出
                out1 = (input_bit ^ s1 ^ s0) & 1
                out2 = (input_bit ^ s0) & 1
                output = [out1, out2]
                
                # 计算汉明距离
                hamming_dist = np.sum(symbols[step] != output)
                
                # 下一状态
                next_state = (input_bit << 1) | s1
                
                # 更新路径度量
                new_metric = path_metrics[state] + hamming_dist
                if new_metric < new_metrics[next_state]:
                    new_metrics[next_state] = new_metric
                    trellis[next_state][step] = (state, input_bit)
        
        path_metrics = new_metrics
    
    # 回溯，从最后的状态 0 开始（因为有尾比特保证回到全零）
    decoded = []
    current_state = 0  # 尾比特使最终状态为 0
    
    for step in range(num_steps - 1, -1, -1):
        prev_state, input_bit = trellis[current_state][step]
        decoded.append(input_bit)
        current_state = prev_state
    
    # 反转得到正确顺序
    decoded = decoded[::-1]
    
    # 去掉尾比特对应的 2 个译码输出（最后 2 位是尾比特 0 产生的）
    return np.array(decoded[:-2], dtype=int)
    # raise NotImplementedError('选做：请实现 Viterbi 硬判决译码')


def run_coding_demo():
    """运行 Part 1 演示并生成 BER 曲线。"""
    print('=' * 60)
    print('Part 1：信道编码实验')
    print('=' * 60)

    error_probabilities = np.array([0.001, 0.003, 0.01, 0.03, 0.06, 0.1])
    uncoded_ber = []
    coded_ber = []

    try:
        bits = generate_bits(4000, seed=2026)
        bits = bits[: len(bits) // 4 * 4]
        encoded = hamming74_encode(bits)

        for index, probability in enumerate(error_probabilities):
            uncoded_rx = binary_symmetric_channel(bits, probability, seed=100 + index)
            encoded_rx = binary_symmetric_channel(encoded, probability, seed=200 + index)
            decoded = hamming74_decode(encoded_rx)
            uncoded_ber.append(calculate_ber(bits, uncoded_rx))
            coded_ber.append(calculate_ber(bits, decoded))

        plot_ber_curve(
            error_probabilities,
            {'未编码': uncoded_ber, 'Hamming(7,4)': coded_ber},
            'Hamming(7,4) 编码前后 BER 对比',
            'coding_ber_curve.png',
        )
        print('✅ 已生成 results/coding_ber_curve.png')
    except NotImplementedError as error:
        print(f'⏸️ 尚未完成核心函数：{error}')
    except Exception as error:
        print(f'❌ Part 1 运行失败：{error}')


if __name__ == '__main__':
    run_coding_demo()
