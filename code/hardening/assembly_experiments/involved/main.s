	.file	"main.cpp"
	.text
	.p2align 4,,15
	.type	_ZL20select_int_nobarrierbii.constprop.0, @function
_ZL20select_int_nobarrierbii.constprop.0:
.LFB2680:
	.cfi_startproc
	movl	%edi, %eax
	movzbl	%dil, %edi
	xorl	$1, %eax
	imull	$89, %edi, %edi
	movzbl	%al, %eax
	imull	$42, %eax, %eax
	addl	%edi, %eax
	ret
	.cfi_endproc
.LFE2680:
	.size	_ZL20select_int_nobarrierbii.constprop.0, .-_ZL20select_int_nobarrierbii.constprop.0
	.p2align 4,,15
	.globl	_Z11logic_chainv
	.type	_Z11logic_chainv, @function
_Z11logic_chainv:
.LFB2350:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	call	rand@PLT
	movl	%eax, %ebx
	call	rand@PLT
	orl	%ebx, %eax
	popq	%rbx
	.cfi_def_cfa_offset 8
	andl	$1, %eax
	ret
	.cfi_endproc
.LFE2350:
	.size	_Z11logic_chainv, .-_Z11logic_chainv
	.section	.text.startup,"ax",@progbits
	.p2align 4,,15
	.globl	main
	.type	main, @function
main:
.LFB2351:
	.cfi_startproc
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	xorl	%edi, %edi
	call	time@PLT
	movl	%eax, %edi
	call	srand@PLT
	call	_Z11logic_chainv
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
	movzbl	%al, %edi
	jmp	_ZL20select_int_nobarrierbii.constprop.0
	.cfi_endproc
.LFE2351:
	.size	main, .-main
	.ident	"GCC: (Debian 8.3.0-6) 8.3.0"
	.section	.note.GNU-stack,"",@progbits
