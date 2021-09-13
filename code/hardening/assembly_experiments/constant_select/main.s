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
.LFB2328:
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
	movl	%eax, -4(%rbp)
	movzbl	-5(%rbp), %eax
	movl	-4(%rbp), %edx
	movl	$0, %esi
	movl	%eax, %edi
	call	_Z6selectIiET_bS0_S0_
	testl	%eax, %eax
	setne	%al
	movb	%al, -5(%rbp)
	movzbl	-5(%rbp), %eax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2328:
	.size	main, .-main
	.type	_Z6selectIiET_bS0_S0_, @function
_Z6selectIiET_bS0_S0_:
.LFB2512:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r12
	pushq	%rbx
	subq	$16, %rsp
	.cfi_offset 12, -24
	.cfi_offset 3, -32
	movl	%edi, %eax
	movl	%esi, -24(%rbp)
	movl	%edx, -28(%rbp)
	movb	%al, -20(%rbp)
	movzbl	-20(%rbp), %eax
	movl	%eax, %edi
	call	_Z13value_barrierIbET_S0_
	movzbl	%al, %ebx
	movl	-24(%rbp), %eax
	movl	%eax, %edi
	call	_Z13value_barrierIiET_S0_
	imull	%eax, %ebx
	movzbl	-20(%rbp), %eax
	xorl	$1, %eax
	movzbl	%al, %eax
	movl	%eax, %edi
	call	_Z13value_barrierIbET_S0_
	movzbl	%al, %r12d
	movl	-28(%rbp), %eax
	movl	%eax, %edi
	call	_Z13value_barrierIiET_S0_
	imull	%r12d, %eax
	addl	%ebx, %eax
	addq	$16, %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2512:
	.size	_Z6selectIiET_bS0_S0_, .-_Z6selectIiET_bS0_S0_
	.type	_Z13value_barrierIbET_S0_, @function
_Z13value_barrierIbET_S0_:
.LFB2570:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, %eax
	movb	%al, -20(%rbp)
	movzbl	-20(%rbp), %eax
	movb	%al, -1(%rbp)
	movzbl	-1(%rbp), %eax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2570:
	.size	_Z13value_barrierIbET_S0_, .-_Z13value_barrierIbET_S0_
	.type	_Z13value_barrierIiET_S0_, @function
_Z13value_barrierIiET_S0_:
.LFB2571:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, -4(%rbp)
	movl	-4(%rbp), %eax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2571:
	.size	_Z13value_barrierIiET_S0_, .-_Z13value_barrierIiET_S0_
	.ident	"GCC: (Debian 8.3.0-6) 8.3.0"
	.section	.note.GNU-stack,"",@progbits
