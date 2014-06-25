#include "render_opts.h"
#include "fullviewer.h"
#include "qexpandbox.h"
#include "err_msgs.h"
#include <QRadioButton>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
#include <QSlider>
#include <QLineEdit>
#include <QCheckBox>

/* upper bound of all sliders' range */
#define SLDR_RANGE 200
#define ONE_BY_SLDR_RANGE 0.005

RenderOpts::RenderOpts (const int my_width, const int working_mode, bool &was_error): QWidget(NULL) {    
	QWidget *tab;

	setFixedWidth(my_width);

	tabs=new(std::nothrow) QExpandBox(NULL,my_width-15,25);
	if (tabs==NULL) { was_error=true; return; }

	tab=new(std::nothrow) QWidget(NULL);
	if (tab==NULL) { was_error=true; return; }
	if (tabs->addItem(tr("Shape of \"points\""),tab)==QExpandBox::Error) { was_error=true; return; }
	repr_cubes=new(std::nothrow) QRadioButton(tab);
	repr_spheres=new(std::nothrow) QRadioButton(tab);
	repr_spheres_info=new(std::nothrow) QPushButton(tab);
	repr_lights=new(std::nothrow) QRadioButton(tab);
	repr_lights_info=new(std::nothrow) QPushButton(tab);
	if ((repr_cubes==NULL) || (repr_spheres==NULL) || (repr_spheres_info==NULL) || 
		(repr_lights==NULL) || (repr_lights_info==NULL)) { was_error=true; return; }

	tab=new(std::nothrow) QWidget(NULL);
	if (tab==NULL) { was_error=true; return; }
	depth_ctr_ind=tabs->addItem(tr("\"Depth constraint\""),tab);
	if (depth_ctr_ind==QExpandBox::Error) { was_error=true; return; }
	depth_constr=new(std::nothrow) QSpinBox(tab);
	depth_info=new(std::nothrow) QPushButton(tab);
	if ((depth_constr==NULL) || (depth_info==NULL)) { was_error=true; return; }

	tab=new(std::nothrow) QWidget(NULL);
	if (tab==NULL) { was_error=true; return; }
	clr_stch_ind=tabs->addItem(tr("Color stretching"),tab);
	if (clr_stch_ind==QExpandBox::Error) { was_error=true; return; }
	clr_stch_info=new(std::nothrow) QPushButton(tab);
	clr_stretching_min=new(std::nothrow) QSlider(tab);
	l_clr_stretch1=new(std::nothrow) QLabel(tab);
	clr_stch_min=new(std::nothrow) QLineEdit(tab);
	clr_stretching_max=new(std::nothrow) QSlider(tab);
	l_clr_stretch2=new(std::nothrow) QLabel(tab);
	clr_stch_max=new(std::nothrow) QLineEdit(tab);
	if ((clr_stch_info==NULL) || (clr_stretching_min==NULL) || 
		(l_clr_stretch1==NULL) || (clr_stch_min==NULL) || (clr_stretching_max==NULL) || 
		(l_clr_stretch2==NULL) || (clr_stch_max==NULL)) { was_error=true; return; }

	tab=new(std::nothrow) QWidget(NULL);
	if (tab==NULL) { was_error=true; return; }
	vol_bld_ind=tabs->addItem(tr("Volume building"),tab);
	if (vol_bld_ind==QExpandBox::Error) { was_error=true; return; }
	vol_bld_mode=new(std::nothrow) QCheckBox(tab);
	vol_bld_info=new(std::nothrow) QPushButton(tab);
	l_vol_build01=new(std::nothrow) QLabel(tab);
	line1=new(std::nothrow) QLabel(tab);
	vol_building_min1=new(std::nothrow) QSlider(tab);
	l_vol_build11=new(std::nothrow) QLabel(tab);
	vol_bld_min1=new(std::nothrow) QLineEdit(tab);
	vol_building_max1=new(std::nothrow) QSlider(tab);
	l_vol_build12=new(std::nothrow) QLabel(tab);
	vol_bld_max1=new(std::nothrow) QLineEdit(tab);
	if ((vol_bld_mode==NULL) || (vol_bld_info==NULL) || (l_vol_build01==NULL) || (line1==NULL) || 
		(vol_building_min1==NULL) || (l_vol_build11==NULL) || (vol_bld_min1==NULL) || (vol_building_max1==NULL) || 
		(l_vol_build12==NULL) || (vol_bld_max1==NULL)) { was_error=true; return; }
	if (working_mode!=0)
	{
		l_vol_build02=new(std::nothrow) QLabel(tab);
		line2=new(std::nothrow) QLabel(tab);
		vol_building_min2=new(std::nothrow) QSlider(tab);
		l_vol_build21=new(std::nothrow) QLabel(tab);
		vol_building_max2=new(std::nothrow) QSlider(tab);
		l_vol_build22=new(std::nothrow) QLabel(tab);
		if ((l_vol_build02==NULL) || (line2==NULL) || (vol_building_min2==NULL) || (l_vol_build21==NULL) || 
			(vol_building_max2==NULL) || (l_vol_build22==NULL)) { was_error=true; return; }
	}
	else
	{
		// second part of this interface is not necessary
		// because only one matrix with values exists
		vol_building_min2=NULL;
		vol_building_max2=NULL;
		l_vol_build02=NULL;
		line2=NULL;
		l_vol_build21=NULL;
		l_vol_build22=NULL;
	}
	vol_bld_min2=new(std::nothrow) QLineEdit(tab);
	vol_bld_max2=new(std::nothrow) QLineEdit(tab);

	was_error=false;
}

void RenderOpts::Init (FullViewer *par) {    
	const QSize help_size(25,20);
	const QString line("______________________");
	QWidget *tab;

	parent=par;
	this->setParent(par);

	QFont font1(font());
	font1.setPointSize(10);
	tabs->setParent(this);
	tabs->setFont(font1);

	tab=tabs->widget(0u);
	tab->setParent(this);
	tab->setFixedHeight(85);
	repr_cubes->setFixedSize(126,20);
	repr_cubes->move(0,0);
	repr_cubes->setText(tr("cubes"));
	repr_cubes->setChecked(true); // must be so!!
	connect(repr_cubes,SIGNAL(clicked(bool)),this,SLOT(ChangePtRepr(void)));
	repr_spheres->setFixedSize(repr_cubes->size());
	repr_spheres->move(repr_cubes->x(),repr_cubes->y()+30);
	repr_spheres->setText(tr("spheres, type 1"));
	repr_spheres->setChecked(false);
	connect(repr_spheres,SIGNAL(clicked(bool)),this,SLOT(ChangePtRepr(void)));
	repr_spheres_info->setFixedSize(help_size);
	repr_spheres_info->move(repr_spheres->x()+repr_spheres->width(),repr_spheres->y());
	repr_spheres_info->setText(tr("?"));
	connect(repr_spheres_info,SIGNAL(clicked()),this,SLOT(ShowReprSpheresInfo(void)));
	repr_lights->setFixedSize(repr_spheres->size());
	repr_lights->move(repr_cubes->x(),repr_spheres->y()+30);
	repr_lights->setText(tr("spheres, type 2"));
	repr_lights->setChecked(false);
	connect(repr_lights,SIGNAL(clicked(bool)),this,SLOT(ChangePtRepr(void)));
	repr_lights_info->setFixedSize(help_size);
	repr_lights_info->move(repr_lights->x()+repr_lights->width(),repr_lights->y());
	repr_lights_info->setText(tr("?"));
	connect(repr_lights_info,SIGNAL(clicked()),this,SLOT(ShowReprLightsInfo(void)));

	tab=tabs->widget(depth_ctr_ind);
	tab->setParent(this);
	tab->setFixedHeight(50);
	depth_constr->setRange(0,(parent->x_num>parent->y_num)? ((parent->x_num>parent->z_num)? parent->x_num : 
																							parent->z_num) : 
	((parent->y_num>parent->z_num)? parent->y_num : parent->z_num));
	depth_constr->setSingleStep(1);
	depth_constr->setValue(depth_constr->maximum()); // initial value - no constraint
	depth_constr->move(20,10);
	connect(depth_constr,SIGNAL(valueChanged(int)),this,SLOT(SetDepthConstraint(void)));
	depth_info->setFixedSize(help_size);
	depth_info->setText(tr("?"));
	depth_info->move(depth_constr->x()+depth_constr->width()-30,depth_constr->y()+
																(depth_constr->height()-help_size.height())/2);
	connect(depth_info,SIGNAL(clicked()),this,SLOT(ShowDepthConstraintInfo(void)));

	tab=tabs->widget(clr_stch_ind);
	tab->setParent(this);
	tab->setFixedHeight(135);
	tabs->setItemEnabled(clr_stch_ind,false);
	clr_stch_info->setFixedSize(help_size);
	clr_stch_info->move((tabs->width())>>1u,0);
	clr_stch_info->setText(tr("?"));
	connect(clr_stch_info,SIGNAL(clicked()),this,SLOT(ShowClrStretchingInfo(void)));
	clr_stretching_min->move(0,clr_stch_info->y()+help_size.height()+2);
	clr_stretching_min->setFixedSize(this->width()-2,20);
	clr_stretching_min->setOrientation(Qt::Horizontal);
	clr_stretching_min->setTickPosition(QSlider::NoTicks);
	clr_stretching_min->setRange(0,SLDR_RANGE);
	clr_stretching_min->setSingleStep(1);
	clr_stretching_min->setSliderPosition(clr_stretching_min->minimum());
	connect(clr_stretching_min,SIGNAL(valueChanged(int)),this,SLOT(AdjustClrStretchingMin(const int)));
	l_clr_stretch1->setFixedSize(37,20);
	l_clr_stretch1->setText(tr("min:"));
	l_clr_stretch1->move(5,clr_stretching_min->y()+clr_stretching_min->height()+5);
	clr_stch_min->setReadOnly(true);
	clr_stch_min->setAlignment(Qt::AlignCenter);
	clr_stch_min->setFixedSize(100,30);
	clr_stch_min->move(l_clr_stretch1->x()+40,l_clr_stretch1->y()-5);
	clr_stch_min->setText("-inf");
	clr_stretching_max->move(clr_stretching_min->x(),clr_stch_min->y()+clr_stch_min->height()+5);
	clr_stretching_max->setFixedSize(clr_stretching_min->size());
	clr_stretching_max->setOrientation(Qt::Horizontal);
	clr_stretching_max->setTickPosition(QSlider::NoTicks);
	clr_stretching_max->setRange(clr_stretching_min->minimum(),clr_stretching_min->maximum());
	clr_stretching_max->setSingleStep(1);
	clr_stretching_max->setSliderPosition(clr_stretching_max->maximum());
	connect(clr_stretching_max,SIGNAL(valueChanged(int)),this,SLOT(AdjustClrStretchingMax(const int)));
	l_clr_stretch2->setFixedSize(l_clr_stretch1->size());
	l_clr_stretch2->setText(tr("max:"));
	l_clr_stretch2->move(l_clr_stretch1->x(),clr_stretching_max->y()+clr_stretching_max->height()+5);
	clr_stch_max->setReadOnly(true);
	clr_stch_max->setAlignment(Qt::AlignCenter);
	clr_stch_max->setFixedSize(clr_stch_min->size());
	clr_stch_max->move(clr_stch_min->x(),l_clr_stretch2->y()-5);
	clr_stch_max->setText("+inf");

	tab=tabs->widget(vol_bld_ind);
	tab->setParent(this);
	tabs->setItemEnabled(vol_bld_ind,false);
	vol_bld_mode->setTristate(false);
	vol_bld_mode->move(0,0);
	vol_bld_mode->setFixedHeight(help_size.height());
	vol_bld_mode->setText(tr("enable mode"));
	vol_bld_mode->setCheckState(Qt::Unchecked);
	connect(vol_bld_mode,SIGNAL(stateChanged(int)),this,SLOT(EnterVolBuildingMode(const int)));
	vol_bld_info->setFixedSize(help_size);
	vol_bld_info->move(vol_bld_mode->x()+vol_bld_mode->text().length()*10+5,0);
	vol_bld_info->setText(tr("?"));
	connect(vol_bld_info,SIGNAL(clicked()),this,SLOT(ShowVolBuildingInfo(void)));
	l_vol_build01->setFixedHeight(25);
	l_vol_build01->setEnabled(false);
	l_vol_build01->move(0,vol_bld_info->y()+help_size.height()+2);
	line1->setEnabled(false);
	line1->setText(line);
	line1->setFixedHeight(20);
	line1->move(l_vol_build01->x(),l_vol_build01->y()+10);
	vol_building_min1->setEnabled(false);
	vol_building_min1->move(0,line1->y()+line1->height()+5);
	vol_building_min1->setFixedSize(this->width()-2,20);
	vol_building_min1->setOrientation(Qt::Horizontal);
	vol_building_min1->setTickPosition(QSlider::NoTicks);
	vol_building_min1->setRange(clr_stretching_max->minimum(),clr_stretching_max->maximum());
	vol_building_min1->setSingleStep(1);
	vol_building_min1->setSliderPosition(vol_building_min1->minimum());
	connect(vol_building_min1,SIGNAL(valueChanged(int)),this,SLOT(AdjustVolBuildingMin1(const int)));
	l_vol_build11->setEnabled(false);
	l_vol_build11->setFixedSize(37,20);
	l_vol_build11->setText(tr("min:"));
	l_vol_build11->move(5,vol_building_min1->y()+vol_building_min1->height()+5);
	vol_bld_min1->setReadOnly(true);
	vol_bld_min1->setAlignment(Qt::AlignCenter);
	vol_bld_min1->setEnabled(false);
	vol_bld_min1->setFixedSize(100,30);
	vol_bld_min1->move(l_vol_build11->x()+40,l_vol_build11->y()-5);
	vol_bld_min1->setText("-inf");
	vol_building_max1->setEnabled(false);
	vol_building_max1->move(vol_building_min1->x(),vol_bld_min1->y()+vol_bld_min1->height()+5);
	vol_building_max1->setFixedSize(vol_building_min1->size());
	vol_building_max1->setOrientation(Qt::Horizontal);
	vol_building_max1->setTickPosition(QSlider::NoTicks);
	vol_building_max1->setRange(vol_building_min1->minimum(),vol_building_min1->maximum());
	vol_building_max1->setSingleStep(1);
	vol_building_max1->setSliderPosition(vol_building_max1->maximum());
	connect(vol_building_max1,SIGNAL(valueChanged(int)),this,SLOT(AdjustVolBuildingMax1(const int)));
	l_vol_build12->setEnabled(false);
	l_vol_build12->setFixedSize(l_vol_build11->size());
	l_vol_build12->setText(tr("max:"));
	l_vol_build12->move(l_vol_build11->x(),vol_building_max1->y()+vol_building_max1->height()+5);
	vol_bld_max1->setReadOnly(true);
	vol_bld_max1->setAlignment(Qt::AlignCenter);
	vol_bld_max1->setEnabled(false);
	vol_bld_max1->setFixedSize(vol_bld_min1->size());
	vol_bld_max1->move(vol_bld_min1->x(),l_vol_build12->y()-5);
	vol_bld_max1->setText("+inf");
	if (parent->working_mode!=0)
	{
		l_vol_build01->setText(tr("Values-1:"));
		tab->setFixedHeight(340);
		l_vol_build02->setEnabled(false);
		l_vol_build02->setText(tr("Values-2:"));
		l_vol_build02->setFixedHeight(l_vol_build01->height());
		l_vol_build02->move(l_vol_build01->x(),vol_bld_max1->y()+vol_bld_max1->height()+10);
		line2->setEnabled(false);
		line2->setText(line);
		line2->setFixedHeight(line1->height());
		line2->move(l_vol_build02->x(),l_vol_build02->y()+10);
		vol_building_min2->setEnabled(false);
		vol_building_min2->move(0,line2->y()+line2->height()+5);
		vol_building_min2->setFixedSize(this->width()-2,20);
		vol_building_min2->setOrientation(Qt::Horizontal);
		vol_building_min2->setTickPosition(QSlider::NoTicks);
		vol_building_min2->setRange(clr_stretching_max->minimum(),clr_stretching_max->maximum());
		vol_building_min2->setSingleStep(1);
		vol_building_min2->setSliderPosition(vol_building_min2->minimum());
		connect(vol_building_min2,SIGNAL(valueChanged(int)),this,SLOT(AdjustVolBuildingMin2(const int)));
		l_vol_build21->setEnabled(false);
		l_vol_build21->setFixedSize(37,20);
		l_vol_build21->setText(tr("min:"));
		l_vol_build21->move(5,vol_building_min2->y()+vol_building_min2->height()+5);
		vol_bld_min2->setReadOnly(true);
		vol_bld_min2->setAlignment(Qt::AlignCenter);
		vol_bld_min2->setEnabled(false);
		vol_bld_min2->setFixedSize(100,30);
		vol_bld_min2->move(l_vol_build21->x()+40,l_vol_build21->y()-5);
		vol_bld_min2->setText("-inf");
		vol_building_max2->setEnabled(false);
		vol_building_max2->move(vol_building_min2->x(),vol_bld_min2->y()+vol_bld_min2->height()+5);
		vol_building_max2->setFixedSize(vol_building_min2->size());
		vol_building_max2->setOrientation(Qt::Horizontal);
		vol_building_max2->setTickPosition(QSlider::NoTicks);
		vol_building_max2->setRange(vol_building_min2->minimum(),vol_building_min2->maximum());
		vol_building_max2->setSingleStep(1);
		vol_building_max2->setSliderPosition(vol_building_max2->maximum());
		connect(vol_building_max2,SIGNAL(valueChanged(int)),this,SLOT(AdjustVolBuildingMax2(const int)));
		l_vol_build22->setEnabled(false);
		l_vol_build22->setFixedSize(l_vol_build21->size());
		l_vol_build22->setText(tr("max:"));
		l_vol_build22->move(l_vol_build21->x(),vol_building_max2->y()+vol_building_max2->height()+5);
		vol_bld_max2->setReadOnly(true);
		vol_bld_max2->setAlignment(Qt::AlignCenter);
		vol_bld_max2->setEnabled(false);
		vol_bld_max2->setFixedSize(vol_bld_min2->size());
		vol_bld_max2->move(vol_bld_min2->x(),l_vol_build22->y()-5);
		vol_bld_max2->setText("+inf");
	}
	else
	{
		// second part of this interface is not necessary
		// because only one matrix with values exists
		l_vol_build01->setText(tr("Values:"));
		tab->setFixedHeight(170);
		vol_bld_min2->hide();
		vol_bld_min2->setText(QString());
		vol_bld_max2->hide();
		vol_bld_max2->setText(QString());
	}

	const unsigned int n=tabs->count();
	int h=(int)(n*25u);
	for (unsigned int i=0u; i!=n; ++i)
		h+=tabs->widget(i)->height();
	setFixedHeight(h);
}

void RenderOpts::ChangePtRepr (void) const {
	if (repr_cubes->isChecked())
		parent->ChangePtRepr(CUBES);
	else
		if (repr_spheres->isChecked())
			parent->ChangePtRepr(SPHERES);
		else
			if (repr_lights->isChecked())
				parent->ChangePtRepr(LIGHTS);
}

void RenderOpts::SetDepthConstraint (void) const { parent->SetDepthConstraint(depth_constr->value()); }

void RenderOpts::AdjustClrStretchingMin (const int val) {
	if (parent->first_render) return;
	const int max_val=clr_stretching_max->value();
	if (val>=max_val)
		clr_stretching_min->setSliderPosition(max_val-1);
	else
	{
		const double new_min=parent->min_val1+(parent->max_val1-parent->min_val1)*((double)val)*ONE_BY_SLDR_RANGE;
		const double new_max=parent->min_val1+(parent->max_val1-parent->min_val1)*((double)max_val)*ONE_BY_SLDR_RANGE;
		clr_stch_min->setText(QString::number(new_min));
		parent->ReFillMatrix(new_min,new_max);
		parent->RenderBox();
	}
}

void RenderOpts::AdjustClrStretchingMax (const int val) {
	if (parent->first_render) return;
	const int min_val=clr_stretching_min->value();
	if (val<=min_val)
		clr_stretching_max->setSliderPosition(min_val+1);
	else
	{
		const double new_min=parent->min_val1+(parent->max_val1-parent->min_val1)*((double)min_val)*ONE_BY_SLDR_RANGE;
		const double new_max=parent->min_val1+(parent->max_val1-parent->min_val1)*((double)val)*ONE_BY_SLDR_RANGE;
		clr_stch_max->setText(QString::number(new_max));
		parent->ReFillMatrix(new_min,new_max);
		parent->RenderBox();
	}
}

void RenderOpts::EnterVolBuildingMode (const int state) {
	const bool checked=(state==Qt::Checked);

	vol_building_min1->setEnabled(checked);
	vol_building_max1->setEnabled(checked);
	l_vol_build11->setEnabled(checked);
	l_vol_build12->setEnabled(checked);
	vol_bld_min1->setEnabled(checked);
	vol_bld_max1->setEnabled(checked);
	if (vol_building_min2!=NULL)
	{
		vol_building_min2->setEnabled(checked);
		vol_building_max2->setEnabled(checked);
		l_vol_build21->setEnabled(checked);
		l_vol_build22->setEnabled(checked);
		vol_bld_min2->setEnabled(checked);
		vol_bld_max2->setEnabled(checked);
	}

	if (parent->renderer->ToggleVolumeMode(checked))
	{
		tabs->setItemEnabled(depth_ctr_ind,!checked); // when 'checked' is true, disable 'depth constraint'
		if (checked)
			parent->AdjustMatrix(vol_bld_min1->text().toDouble(),vol_bld_max1->text().toDouble(),
								 vol_bld_min2->text().toDouble(),vol_bld_max2->text().toDouble());
		else parent->RenderBox();
	}
	else
		emit parent->SendMessToLog(MainWindow::Error,FullViewer::my_sign,
								   ErrMsgs::ToString(NV::RenderError,1,&(parent->renderer->GetError())));
}

void RenderOpts::AdjustVolBuildingMin1 (const int val) {
	const int max_val=vol_building_max1->value();
	if (val>=max_val)
		vol_building_min1->setValue(max_val-1);
	else
	{
		const double new_min=parent->min_val1+(parent->max_val1-parent->min_val1)*((double)val)*ONE_BY_SLDR_RANGE;
		vol_bld_min1->setText(QString::number(new_min));
		parent->AdjustMatrix(new_min,vol_bld_max1->text().toDouble(),vol_bld_min2->text().toDouble(),
							 vol_bld_max2->text().toDouble());
	}
}

void RenderOpts::AdjustVolBuildingMax1 (const int val) {
	const int min_val=vol_building_min1->value();
	if (val<=min_val)
		vol_building_max1->setValue(min_val+1);
	else
	{
		const double new_max=parent->min_val1+(parent->max_val1-parent->min_val1)*((double)val)*ONE_BY_SLDR_RANGE;
		vol_bld_max1->setText(QString::number(new_max));
		parent->AdjustMatrix(vol_bld_min1->text().toDouble(),new_max,vol_bld_min2->text().toDouble(),
							 vol_bld_max2->text().toDouble());
	}
}

void RenderOpts::AdjustVolBuildingMin2 (const int val) {
	const int max_val=vol_building_max2->value();
	if (val>=max_val)
		vol_building_min2->setValue(max_val-1);
	else
	{
		const double new_min=parent->min_val2+(parent->max_val2-parent->min_val2)*((double)val)*ONE_BY_SLDR_RANGE;
		vol_bld_min2->setText(QString::number(new_min));
		parent->AdjustMatrix(vol_bld_min1->text().toDouble(),vol_bld_max1->text().toDouble(),new_min,
							 vol_bld_max2->text().toDouble());
	}
}

void RenderOpts::AdjustVolBuildingMax2 (const int val) {
	const int min_val=vol_building_min2->value();
	if (val<=min_val)
		vol_building_max2->setValue(min_val+1);
	else
	{
		const double new_max=parent->min_val2+(parent->max_val2-parent->min_val2)*((double)val)*ONE_BY_SLDR_RANGE;
		vol_bld_max2->setText(QString::number(new_max));
		parent->AdjustMatrix(vol_bld_min1->text().toDouble(),vol_bld_max1->text().toDouble(),
							 vol_bld_min2->text().toDouble(),new_max);
	}
}

void RenderOpts::ActivateAll (void) {
	// color stretching
	tabs->setItemEnabled(clr_stch_ind,true);
	clr_stch_min->setText(QString::number(parent->min_val1));
	clr_stch_max->setText(QString::number(parent->max_val1));
	// volume building
	tabs->setItemEnabled(vol_bld_ind,true);
	vol_bld_min1->setText(QString::number(parent->min_val1));
	vol_bld_max1->setText(QString::number(parent->max_val1));
	vol_bld_min2->setText(QString::number(parent->min_val2));
	vol_bld_max2->setText(QString::number(parent->max_val2));
}

void RenderOpts::GetClrStretchingMinMax (double &min_val, double &max_val) {
	min_val=parent->min_val1+(parent->max_val1-parent->min_val1)* \
			static_cast<double>(clr_stretching_min->value())*ONE_BY_SLDR_RANGE;
	max_val=parent->min_val1+(parent->max_val1-parent->min_val1)* \
			static_cast<double>(clr_stretching_max->value())*ONE_BY_SLDR_RANGE;
}

RenderOpts::~RenderOpts () {
	delete repr_cubes;
	delete repr_spheres;
	delete repr_lights;
	delete repr_spheres_info;
	delete repr_lights_info;
	delete depth_info;
	delete depth_constr;
	delete clr_stretching_min;
	delete clr_stretching_max;
	delete l_clr_stretch1;
	delete l_clr_stretch2;
	delete clr_stch_min;
	delete clr_stch_max;
	delete clr_stch_info;
	delete vol_bld_mode;
	delete vol_building_min1;
	delete vol_building_max1;
	delete l_vol_build01;
	delete line1;
	delete l_vol_build11;
	delete l_vol_build12;
	if (vol_building_min2!=NULL)
	{
		delete vol_building_min2;
		delete vol_building_max2;
		delete l_vol_build02;
		delete line2;
		delete l_vol_build21;
		delete l_vol_build22;
	}
	delete vol_bld_min1;
	delete vol_bld_max1;
	delete vol_bld_min2;
	delete vol_bld_max2;
	delete vol_bld_info;
	const unsigned int n=tabs->count();
	for (unsigned int i=0u; i<n; ++i)
		delete tabs->widget(i);
	delete tabs;
}

