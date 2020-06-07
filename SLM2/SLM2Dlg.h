
// SLM2Dlg.h : header file
//
#include "Hologram.h"
#pragma once


// CSLM2Dlg dialog
class CSLM2Dlg : public CDialogEx
{
// Construction
	public:
		CSLM2Dlg(CWnd* pParent = nullptr);	// standard constructor
	

	// Dialog Data
	#ifdef AFX_DESIGN_TIME
		enum { IDD = IDD_SLM2_DIALOG };
	#endif

	protected:
		virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	// Implementation
	protected:
		HICON m_hIcon;
		CEdit test;
		// Generated message map functions
		virtual BOOL OnInitDialog();
		//afx_msg void OnPaint();
		afx_msg HCURSOR OnQueryDragIcon();
		DECLARE_MESSAGE_MAP()
		public:
		afx_msg void OnBnClickedButton1();
		afx_msg BOOL PreTranslateMessage(MSG* pMsg);//so that it doesn't exit if you hit enter
		// you need the doDataExchange function if you want these variables for 
		// edit boxes that were made with the wizard.
		double m_edit1C;
		double m_edit2C;
		double m_edit3C;
		double m_edit4C;
		Hologram SLM;
};
