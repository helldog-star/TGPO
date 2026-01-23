import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def unify_vocab_to_tokenizer_max(
    student_model_path: str,
    teacher_model_path: str,
    output_dir: str = None,
    unify_special_tokens: bool = True,
    # add_think_to_chat_template: bool = False,
):
    """
    统一 Student 和 Teacher 的词表大小到 tokenizer 的最大值，并统一特殊 token
    
    这是最安全的方案，因为：
    1. tokenizer 决定了实际能生成的 token
    2. 模型的 embedding 层应该能容纳 tokenizer 生成的所有 token
    3. 避免 tokenizer 生成的 token 超出模型范围
    4. 统一特殊 token 确保 RLHF 训练的一致性
    
    Args:
        student_model_path: Student 模型路径
        teacher_model_path: Teacher 模型路径
        output_dir: 输出目录（如果为 None，则在原路径后添加 -aligned）
        unify_special_tokens: 是否统一特殊 token（EOS, PAD, BOS, UNK）
    """
    
    print("=" * 80)
    print("模型对齐工具 (词表 + 特殊 Token)")
    print("=" * 80)
    
    # ========== 第一步：加载 tokenizer ==========
    print("\n📖 加载 tokenizer...")
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_path)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
    
    student_tokenizer_vocab = len(student_tokenizer)
    teacher_tokenizer_vocab = len(teacher_tokenizer)
    
    print(f"  Student tokenizer vocab: {student_tokenizer_vocab}")
    print(f"  Teacher tokenizer vocab: {teacher_tokenizer_vocab}")
    
    # ========== 第二步：确定目标词表大小 ==========
    target_vocab_size = max(student_tokenizer_vocab, teacher_tokenizer_vocab)
    print(f"\n🎯 目标词表大小: {target_vocab_size}")
    print(f"   (基于 tokenizer 的最大值)")
    
    # ========== 第三步：加载模型 ==========
    print("\n🔄 加载模型...")
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    
    print(f"  Student embedding 层: {student_model.get_input_embeddings().weight.shape[0]}")
    print(f"  Teacher embedding 层: {teacher_model.get_input_embeddings().weight.shape[0]}")
    
    # ========== 第四步：调整词表大小 ==========
    print(f"\n🔧 调整模型词表大小到 {target_vocab_size}...")
    
    if student_model.config.vocab_size != target_vocab_size:
        print(f"  调整 Student: {student_model.config.vocab_size} -> {target_vocab_size}")
        student_model.resize_token_embeddings(target_vocab_size)
    else:
        print(f"  Student 已匹配")
    
    if teacher_model.config.vocab_size != target_vocab_size:
        print(f"  调整 Teacher: {teacher_model.config.vocab_size} -> {target_vocab_size}")
        teacher_model.resize_token_embeddings(target_vocab_size)
    else:
        print(f"  Teacher 已匹配")
    
    # ========== 第五步：统一特殊 Token ==========
    if unify_special_tokens:
        print(f"\n🔧 统一特殊 Token...")
        _unify_special_tokens(
            student_tokenizer, student_model,
            teacher_tokenizer, teacher_model, False
        )
    
    # ========== 第六步：验证 ==========
    print(f"\n✅ 验证调整结果...")
    print(f"  Student:")
    print(f"    - config.vocab_size: {student_model.config.vocab_size}")
    print(f"    - embedding 层: {student_model.get_input_embeddings().weight.shape[0]}")
    print(f"    - tokenizer: {len(student_tokenizer)}")
    print(f"    - eos_token_id: {student_tokenizer.eos_token_id}")
    print(f"    - pad_token_id: {student_tokenizer.pad_token_id}")
    
    print(f"  Teacher:")
    print(f"    - config.vocab_size: {teacher_model.config.vocab_size}")
    print(f"    - embedding 层: {teacher_model.get_input_embeddings().weight.shape[0]}")
    print(f"    - tokenizer: {len(teacher_tokenizer)}")
    print(f"    - eos_token_id: {teacher_tokenizer.eos_token_id}")
    print(f"    - pad_token_id: {teacher_tokenizer.pad_token_id}")
    
    # 验证词表大小
    assert student_model.config.vocab_size == target_vocab_size, \
        f"Student vocab_size 不匹配: {student_model.config.vocab_size} != {target_vocab_size}"
    assert teacher_model.config.vocab_size == target_vocab_size, \
        f"Teacher vocab_size 不匹配: {teacher_model.config.vocab_size} != {target_vocab_size}"
    
    # 验证特殊 token
    if unify_special_tokens:
        assert student_tokenizer.eos_token_id == teacher_tokenizer.eos_token_id, \
            f"EOS token 不匹配: {student_tokenizer.eos_token_id} != {teacher_tokenizer.eos_token_id}"
        assert student_tokenizer.pad_token_id == teacher_tokenizer.pad_token_id, \
            f"PAD token 不匹配: {student_tokenizer.pad_token_id} != {teacher_tokenizer.pad_token_id}"
    
    # ========== 第七步：保存 ==========
    if output_dir is None:
        output_student = student_model_path + "-aligned"
        output_teacher = teacher_model_path + "-aligned"
    else:
        output_student = os.path.join(output_dir, "student-aligned")
        output_teacher = os.path.join(output_dir, "teacher-aligned")

    
    print(f"\n💾 保存模型...")
    os.makedirs(output_student, exist_ok=True)
    os.makedirs(output_teacher, exist_ok=True)
    
    student_model.save_pretrained(output_student)
    student_tokenizer.save_pretrained(output_student)
    
    teacher_model.save_pretrained(output_teacher)
    teacher_tokenizer.save_pretrained(output_teacher)
    
    print(f"  ✅ Student 已保存到: {output_student}")
    print(f"  ✅ Teacher 已保存到: {output_teacher}")
    
    # ========== 第八步：生成诊断报告 ==========
    print(f"\n📊 最终对齐状态报告...")
    print(f"  ┌─────────────┬──────────┬──────────┬──────────┬──────────┐")
    print(f"  │             │ config   │ embedding│ tokenizer│ eos_id   │")
    print(f"  ├─────────────┼──────────┼──────────┼──────────┼──────────┤")
    print(f"  │ Student     │ {student_model.config.vocab_size:8d} │ {student_model.get_input_embeddings().weight.shape[0]:8d} │ {len(student_tokenizer):8d} │ {student_tokenizer.eos_token_id:8d} │")
    print(f"  │ Teacher     │ {teacher_model.config.vocab_size:8d} │ {teacher_model.get_input_embeddings().weight.shape[0]:8d} │ {len(teacher_tokenizer):8d} │ {teacher_tokenizer.eos_token_id:8d} │")
    print(f"  └─────────────┴──────────┴──────────┴──────────┴──────────┘")
    
    print("\n" + "=" * 80)
    print("✅ 模型对齐完成！")
    print("=" * 80)
    
    return output_student, output_teacher


def _unify_special_tokens(
    student_tokenizer,
    student_model,
    teacher_tokenizer,
    teacher_model,
    use_teacher_tokens: bool = True
):
    """
    统一特殊 token（EOS, PAD, BOS, UNK）
    
    Args:
        use_teacher_tokens: 如果 True，使用 Teacher 的特殊 token；否则使用 Student 的
    """
    
    # 定义要统一的特殊 token
    special_tokens_to_unify = [
        'eos_token_id',
        'pad_token_id',
        'bos_token_id',
        'unk_token_id',
    ]
    
    print("\n  📋 当前特殊 Token:")
    print(f"    {'Token':15s} {'Student':10s} {'Teacher':10s} {'Status':10s}")
    print(f"    {'-' * 50}")
    
    # 显示当前状态
    for token_name in special_tokens_to_unify:
        student_val = getattr(student_tokenizer, token_name, None)
        teacher_val = getattr(teacher_tokenizer, token_name, None)
        
        if student_val == teacher_val:
            status = "✅ 匹配"
        else:
            status = "❌ 不匹配"
        
        print(f"    {token_name:15s} {str(student_val):10s} {str(teacher_val):10s} {status:10s}")
    
    # 确定目标 token
    if use_teacher_tokens:
        print(f"\n  🎯 使用 Teacher 的特殊 Token")
        target_tokens = {
            token_name: getattr(teacher_tokenizer, token_name, None)
            for token_name in special_tokens_to_unify
        }
    else:
        print(f"\n  🎯 使用 Student 的特殊 Token")
        target_tokens = {
            token_name: getattr(student_tokenizer, token_name, None)
            for token_name in special_tokens_to_unify
        }
    
    # 更新 Student
    print(f"\n  🔧 更新 Student 特殊 Token:")
    for token_name, target_val in target_tokens.items():
        current_val = getattr(student_tokenizer, token_name, None)
        if current_val != target_val and target_val is not None:
            print(f"    {token_name}: {current_val} -> {target_val}")
            setattr(student_tokenizer, token_name, target_val)
            setattr(student_model.config, token_name, target_val)
    
    # 更新 Teacher
    print(f"\n  🔧 更新 Teacher 特殊 Token:")
    for token_name, target_val in target_tokens.items():
        current_val = getattr(teacher_tokenizer, token_name, None)
        if current_val != target_val and target_val is not None:
            print(f"    {token_name}: {current_val} -> {target_val}")
            setattr(teacher_tokenizer, token_name, target_val)
            setattr(teacher_model.config, token_name, target_val)
    
    print(f"\n  ✅ 特殊 Token 统一完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对齐 Student 与 Teacher 的 tokenizer（词表、特殊 token）")
    parser.add_argument("--student", required=True, help="Student 模型路径")
    parser.add_argument("--teacher", required=True, help="Teacher 模型路径（作为词表与特殊 token 的参考）")
    parser.add_argument("--output-dir", default=None, help="输出根目录；默认在各自路径后加 -aligned")
    parser.add_argument("--no-unify-special-tokens", action="store_true", help="不统一特殊 token")
    args = parser.parse_args()

    unify_vocab_to_tokenizer_max(
        student_model_path=args.student,
        teacher_model_path=args.teacher,
        output_dir=args.output_dir,
        unify_special_tokens=not args.no_unify_special_tokens,
    )