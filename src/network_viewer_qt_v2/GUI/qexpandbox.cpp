/* This file is a part of PARUS project and NOT of 
   Qt Toolkit! If you want to include this file to 
   Qt Toolkit, please mail to pashokkk@bk.ru first. */

#include "qexpandbox.h"

QExpandBox::QExpandBox (QWidget *par, const unsigned int buttons_w, const unsigned int buttons_h): 
  QWidget(par), btn_h(buttons_h) {
	mem_num=4u;
	masters=static_cast<MasterButton**>(malloc(mem_num*sizeof(MasterButton*)));
	slaves=static_cast<QWidget**>(malloc(mem_num*sizeof(QWidget*)));
	number=0u;

	font=new(std::nothrow) QFont();

	QWidget::setFixedSize(buttons_w,0);
}

QExpandBox::~QExpandBox () {
	delete font;
	for (unsigned int i=0u; i<number; ++i)
	delete masters[i];
	free(masters);
	free(slaves);
}

void QExpandBox::setFont (const QFont &fnt) {
	// change font of already created buttons
	const int pt_size=fnt.pointSize();
	const int btn_w=width();
	QString txt;
	int pos;

	for (unsigned int i=0u; i<number; ++i)
	{
		masters[i]->setFont(fnt);
		txt=masters[i]->text();
		if ((txt.length()-2)*pt_size>btn_w)
		{
			// 'txt' is too long for 'masters[i]'
			pos=(btn_w+40)/pt_size-3;
			masters[i]->setText(txt.replace(pos,txt.length()-pos,"..."));
		}
	}
	delete font;
	font=new(std::nothrow) QFont(fnt);
}

unsigned int QExpandBox::addItem (const QString &btn_text, QWidget *wdg) {
	MasterButton *btn=new(std::nothrow) MasterButton(this,number);
	if (btn==NULL) return Error; // not enough memory

	const int btn_w=width();

	btn->setFixedSize(btn_w,btn_h);
	btn->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
	btn->setArrowType(Qt::RightArrow);
	btn->setFont(*font);
	if ((btn_text.length()-2)*font->pointSize()>btn_w)
	{
		// 'btn_text' is too long for 'btn'
		QString txt(btn_text);
		const int i=(btn_w+40)/font->pointSize()-3;
		btn->setText(txt.replace(i,btn_text.length()-i,"..."));
#ifndef QT_NO_TOOLTIP
		btn->setToolTip(btn_text);
#endif
	}
	else btn->setText(btn_text);
	btn->move(0,height());
	btn->show();

	QWidget::setFixedHeight(height()+btn_h);

	wdg->hide();

	if (number==mem_num)
	{
		mem_num=mem_num+(mem_num>>1u); // increase the amount of memory by factor 1.5
		masters=static_cast<MasterButton**>(realloc(masters,mem_num*sizeof(MasterButton*)));
		slaves=static_cast<QWidget**>(realloc(slaves,mem_num*sizeof(QWidget*)));
		if ((masters==NULL) || (slaves==NULL)) return Error; // not enough memory
	}
	masters[number]=btn;
	slaves[number]=wdg;
	++number;
	return (number-1u);
}

void QExpandBox::operate (const unsigned int ind) {
	const int m_x=x(),m_y=y();
	int move_h=static_cast<int>(btn_h*ind+btn_h);
	unsigned int i;
	QWidget *slave;

	for (i=0u; i!=ind; ++i)
	{
		slave=slaves[i];
		if (slave->isVisible()) move_h+=slave->height();
	}
	slave=slaves[ind];
	if (slave->isVisible())
	{
		masters[ind]->setArrowType(Qt::RightArrow);
		slave->hide();

		QWidget::setFixedHeight(height()-slave->height());
	}
	else
	{
		masters[ind]->setArrowType(Qt::DownArrow);
		slave->move(m_x,move_h+m_y);
		slave->show();
		move_h+=slave->height();

		QWidget::setFixedHeight(height()+slave->height());
	}
	for (i=ind+1u; i<number; ++i)
	{
		masters[i]->move(0,move_h);
		move_h+=static_cast<int>(btn_h);
		slave=slaves[i];
		if (slave->isVisible())
		{
			slave->move(m_x,move_h+m_y);
			move_h+=slave->height();
		}
	}
}

