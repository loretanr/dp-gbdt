	.file	"main.cpp"
	.text
	.globl	_Z11logic_chainv
	.type	_Z11logic_chainv, @function
_Z11logic_chainv:
.LFB2347:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	call	rand@PLT
	movl	%eax, %ebx
	call	rand@PLT
	orl	%ebx, %eax
	andl	$1, %eax
	popq	%rbx
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE2347:
	.size	_Z11logic_chainv, .-_Z11logic_chainv
	.globl	main
	.type	main, @function
main:
.LFB2348:
	.cfi_startproc
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	movl	$0, %edi
	call	time@PLT
	movl	%eax, %edi
	call	srand@PLT
	call	_Z11logic_chainv
	movl	%eax, %edx
	xorl	$1, %edx
	movzbl	%dl, %edx
	imull	$42, %edx, %edx
	movzbl	%al, %eax
	leal	(%rax,%rax,4), %eax
	addl	%edx, %eax
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE2348:
	.size	main, .-main
	.ident	"GCC: (Debian 8.3.0-6) 8.3.0"
	.section	.note.GNU-stack,"",@progbits