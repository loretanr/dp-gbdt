	.file	"main.cpp"
	.text
	.section	.rodata
	.type	_ZStL19piecewise_construct, @object
	.size	_ZStL19piecewise_construct, 1
_ZStL19piecewise_construct:
	.zero	1
	.text
	.globl	_Z11logic_chainv
	.type	_Z11logic_chainv, @function
_Z11logic_chainv:
.LFB2326:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	call	rand@PLT
	andl	$1, %eax
	testl	%eax, %eax
	setne	%al
	movb	%al, -1(%rbp)
	call	rand@PLT
	andl	$1, %eax
	testl	%eax, %eax
	setne	%al
	movb	%al, -2(%rbp)
	cmpb	$0, -1(%rbp)
	jne	.L2
	cmpb	$0, -2(%rbp)
	je	.L3
.L2:
	movl	$1, %eax
	jmp	.L4
.L3:
	movl	$0, %eax
.L4:
	movb	%al, -3(%rbp)
	movzbl	-3(%rbp), %eax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2326:
	.size	_Z11logic_chainv, .-_Z11logic_chainv
	.globl	main
	.type	main, @function
main:
.LFB2327:
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
	call	_Z11logic_chainv
	movb	%al, -1(%rbp)
	movl	$5, -8(%rbp)
	movl	$42, -12(%rbp)
	movzbl	-1(%rbp), %eax
	imull	-8(%rbp), %eax
	movl	%eax, %edx
	movzbl	-1(%rbp), %eax
	xorl	$1, %eax
	movzbl	%al, %eax
	imull	-12(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2327:
	.size	main, .-main
	.ident	"GCC: (Debian 8.3.0-6) 8.3.0"
	.section	.note.GNU-stack,"",@progbits
