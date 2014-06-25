/* This file is a part of PARUS project and NOT of 
   Qt Toolkit! If you want to include this file to 
   Qt Toolkit, please mail to pashokkk@bk.ru first. */

#pragma once

#include <QToolButton>

class MasterButton;

class QExpandBox: public QWidget
{
	Q_OBJECT
	
	friend class MasterButton;

  public:
	  static const unsigned int Error=static_cast<const unsigned int>(-1); // addItem() may return this

  private:
	  MasterButton **masters; // control buttons
	  QWidget **slaves; // widgets under their control (one widget per one button)
	  unsigned int number; // number of elements in arrays above
	  unsigned int mem_num; // number of allocated elements
	  
	  const unsigned int btn_h; // height of all buttons
	  QFont *font; // text on all buttons will be written using this font

  public:
	  explicit QExpandBox (QWidget *parent=NULL, const unsigned int buttons_w=0u, const unsigned int buttons_h=0u);
	  
	  virtual ~QExpandBox ();
	  
	  // sets font for text on all buttons (new font overwrites old one)
	  void setFont (const QFont &fnt);
	  
	  // adds pair (button['btn_text'],wdg) to the end of the array;
	  // no existance checking is performed!
	  // returns array index of added item
	  unsigned int addItem (const QString &btn_text, QWidget *wdg);
	  
	  // enables/disables master button with index 'ind'
	  void setItemEnabled (const unsigned int ind, const bool enabled);
	  
	  // returns number of widgets/buttons
	  unsigned int count () const { return number; }
	  
	  // returns widget with index 'ind'
	  QWidget* widget (const unsigned int ind) { return slaves[ind]; }
	  
	  // does "wdg->setParent(masters[ind])"
	  void addButtonChild (const unsigned int ind, QWidget *wdg);

  private:
	  // shows/hides widget with index 'ind'
	  Q_SLOT void operate (const unsigned int ind);
	  
	  Q_DISABLE_COPY(QExpandBox)
	  /* do NOT call these functions! */
	  virtual void setFixedSize (int, int) { Q_ASSERT(0); }
	  virtual void setFixedWidth (int) { Q_ASSERT(0); }
	  virtual void setFixedHeight (int) { Q_ASSERT(0); }
};

class MasterButton: public QToolButton
{
	Q_OBJECT

  private:
	  const unsigned int indx; // index in the array of buttons

  public:
	  MasterButton (QExpandBox *p, unsigned int i): QToolButton(p),indx(i) {
		  connect(this,SIGNAL(clicked()),this,SLOT(operate()));
		  connect(this,SIGNAL(iWasClicked(const unsigned int)),p,SLOT(operate(const unsigned int)));
	  }

  private:
	  Q_SIGNAL void iWasClicked (const unsigned int);
	  Q_SLOT void operate () { emit iWasClicked(indx); }
	  
	  Q_DISABLE_COPY(MasterButton)
};

inline void QExpandBox::setItemEnabled (const unsigned int ind, const bool enabled) {
	if (slaves[ind]->isVisible()) operate(ind); // it hides inactive widget if necessary
	masters[ind]->setEnabled(enabled);
}

inline void QExpandBox::addButtonChild (const unsigned int ind, QWidget *wdg) {
	if (wdg!=NULL) wdg->setParent(masters[ind]);
}

