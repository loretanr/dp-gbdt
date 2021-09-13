	.file	"main.cpp"
	.text
	.section	.rodata
	.type	_ZStL19piecewise_construct, @object
	.size	_ZStL19piecewise_construct, 1
_ZStL19piecewise_construct:
	.zero	1
	.text
	.globl	main
	.type	main, @function
main:
.LFB2326:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movl	$0, %edi
	call	time@PLT
	movl	%eax, %edi
	call	srand@PLT
	call	rand@PLT
	movl	%eax, %ecx
	movl	$818089009, %edx
	movl	%ecx, %eax
	imull	%edx
	sarl	$3, %edx
	movl	%ecx, %eax
	sarl	$31, %eax
	subl	%eax, %edx
	movl	%edx, %eax
	movl	%eax, -4(%rbp)
	movl	-4(%rbp), %eax
	imull	$42, %eax, %eax
	subl	%eax, %ecx
	movl	%ecx, %eax
	movl	%eax, -4(%rbp)
	cmpl	$0, -4(%rbp)
	sete	%al
	movb	%al, -5(%rbp)
	movzbl	-5(%rbp), %eax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2326:
	.size	main, .-main
	.ident	"GCC: (Debian 8.3.0-6) 8.3.0"
	.section	.note.GNU-stack,"",@progbits
