
// SLM2Dlg.cpp : implementation file
//

//#include "pch.h"
#include "framework.h"
#include "SLM2.h"
#include "SLM2Dlg.h"
#include "afxdialogex.h"
#include "externals.h"
#include "resource.h"
#include "my_str.h"
#include "thrower.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CSLM2Dlg dialog



CSLM2Dlg::CSLM2Dlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_SLM2_DIALOG, pParent)
	, m_edit1C()
	, m_edit2C()
	, m_edit3C()
	, m_edit4C()

{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CSLM2Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);

	DDX_Text(pDX, IDC_EDIT1, m_edit1C);
	DDX_Text(pDX, IDC_EDIT2, m_edit2C);
	DDX_Text(pDX, IDC_EDIT3, m_edit3C);
	DDX_Text(pDX, IDC_EDIT4, m_edit4C);
}

BEGIN_MESSAGE_MAP(CSLM2Dlg, CDialogEx)
	//ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_CTLCOLOR()
	ON_BN_CLICKED(IDC_BUTTON1, &CSLM2Dlg::OnBnClickedButton1)
END_MESSAGE_MAP()


// CSLM2Dlg message handlers

BOOL CSLM2Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	SetWindowPos(this, 0,0, 1900, 1100, 0);
	ShowWindow(SW_MAXIMIZE);

	ShowWindow(SW_MINIMIZE);

	// TODO: Add extra initialization here

	return TRUE;  // return TRUE  unless you set the focus to a control
}



// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

/*void CSLM2Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		dc.SetBkColor(_myRGBs["Interactable-Bkgd"]);
		CRect rect;
		GetClientRect(&rect);
		SetBackgroundColor(_myRGBs["Interactable-Bkgd"]);

		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;


		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}*/

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CSLM2Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CSLM2Dlg::OnBnClickedButton1()
{
	// TODO: Add your control notification handler code here
	UpdateData();//otherwise the variables don't change 
	SLM.setNX(256);
	SLM.setNY(256);
	SLM.setActive(128);
	SLM.setSpace(50);//needs to be changed later
	SLM.setDim(10);//
	SLM.populateArrays();
	SLM.gerchbergPhaseLoop(int(m_edit1C));
	SLM.gerchbergAmpLoop(int(m_edit2C));
}



//So that it doesn't exit if you hit enter
BOOL CSLM2Dlg::PreTranslateMessage(MSG* pMsg)
{
	if (pMsg)
	{
		if (pMsg->message == WM_KEYDOWN)
		{
			if (pMsg->wParam == VK_RETURN | pMsg->wParam == VK_ESCAPE)
				pMsg->wParam = NULL;
		}
	}
	// Call the base class PreTranslateMessage. In this example,
	// CRhinoDialog is the base class to CMyModalDialog.
	return CDialog::PreTranslateMessage(pMsg);
}

