#include "renderer_OpenMP.h"
#include <cfloat>
#ifdef _OPENMP
  #include <omp.h>
#endif

void RendererOMP::Init (const unsigned int image_w, const unsigned int image_h, 
						const int num_x, const int num_y, const int num_z) {
	img_w=image_w;
	img_h=image_h;

	point_step=5.0f;
	point_radius=0.4f*point_step;

	cam_x=0.0f;
	cam_y=0.0f;
	cam_z=-500.0f;

	x_num=num_x;
	y_num=num_y;
	z_num=num_z;
	x_size=static_cast<float>(num_x)*point_step;
	y_size=static_cast<float>(num_y)*point_step;
	z_size=static_cast<float>(num_z)*point_step;

	/*const float len=sqrtf(0.2f);
	ox_x=len+len;
	ox_y=0.0f;
	ox_z=len;

	oy_x=0.0f;
	oy_y=-1.0f;
	oy_z=0.0f;

	oz_x=-len;
	oz_y=0.0f;
	oz_z=ox_x;*/
	ox_x=1.0f;
	ox_y=0.0f;
	ox_z=0.0f;

	oy_x=0.0f;
	oy_y=-1.0f;
	oy_z=0.0f;

	oz_x=0.0f;
	oz_y=0.0f;
	oz_z=1.0f;

	bbox_cnt_x=0.0f;
	bbox_cnt_y=0.0f;
	bbox_cnt_z=5200.0f+0.5f*z_size;

	depth_constraint=1u<<20u; // some kind of big number

	clr_minmax=0xff00ff00; // min_green=0, max_green=255, min_red=0, max_red=255

	pt_selection_mode=false;
}

unsigned int RendererOMP::NumberCUs (void) const {
#ifdef _OPENMP
	unsigned int thr;
	#pragma omp parallel
	{
	#pragma omp single
		thr=static_cast<unsigned int>(omp_get_num_threads());
	}
	return thr;
#else
	return 1u;
#endif
}

#define GET_G(clr) (clr & 0xff)
#define GET_R(clr) (clr>>8)

void RendererOMP::RenderBox_cubes (unsigned int *pixels) const {
	const float minus_img_w_2=-static_cast<const float>(img_w>>1u);
	const float minus_img_h_2=-static_cast<const float>(img_h>>1u);
	const float ray_dir_z_init=-minus_img_w_2*2.414213562f;

	/** all computations will be in bounding box coordinate system! **/

	// "left-bottom-near" point is an origin of bounding box coordinate system
	const float ray_pos_x_const=(cam_x-bbox_cnt_x)*ox_x+(cam_y-bbox_cnt_y)*ox_y+(cam_z-bbox_cnt_z)*ox_z+0.5f*x_size,
	ray_pos_y_const=(cam_x-bbox_cnt_x)*oy_x+(cam_y-bbox_cnt_y)*oy_y+(cam_z-bbox_cnt_z)*oy_z+0.5f*y_size,
	ray_pos_z_const=(cam_x-bbox_cnt_x)*oz_x+(cam_y-bbox_cnt_y)*oz_y+(cam_z-bbox_cnt_z)*oz_z+0.5f*z_size;

	#pragma omp parallel
	{
	unsigned int i,j;
	// Do NOT vectorize these variables! The speed will be decreased! Oh, you can check...
	float ray_pos_x,ray_pos_y,ray_pos_z,ray_dir_x,ray_dir_y,ray_dir_z;
	float ttime,t;
	float coord1,coord2;
	int ind_k,ind_l,ind_m;
	int ind_k_init,ind_l_init,ind_m_init;
	int ind_k_step,ind_l_step,ind_m_step;
	float t_max_x,t_max_y,t_max_z;
	float t_delta_x,t_delta_y,t_delta_z;
	float main_color_g,main_color_r;
	float self_intense,this_point_intense_g,this_point_intense_r;
	int r_d_x_ge_0,r_d_y_ge_0,r_d_z_ge_0;
	int mtr_indx;
	int xy_num,yl_num;

	#pragma omp for schedule(dynamic,2) nowait
	for (i=0; i<img_h; ++i)
	{
		for (j=0; j<img_w; ++j)
		{
			/* transform ray coordinates */
			ttime=minus_img_w_2+j;
			t=minus_img_h_2+i;
			ray_dir_x=ttime*ox_x+t*ox_y+ray_dir_z_init*ox_z;
			ray_dir_y=ttime*oy_x+t*oy_y+ray_dir_z_init*oy_z;
			ray_dir_z=ttime*oz_x+t*oz_y+ray_dir_z_init*oz_z;
			t=sqrtf(ray_dir_x*ray_dir_x+ray_dir_y*ray_dir_y+ray_dir_z*ray_dir_z);
			ray_dir_x/=t;
			ray_dir_y/=t;
			ray_dir_z/=t;

			ttime=1.0e+30f;

			/* find intersection of the ray and the main bounding box */
			if ((ray_dir_z<-0.001f) || (ray_dir_z>0.001f))
			{
				t=-ray_pos_z_const/ray_dir_z;
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_y_const+t*ray_dir_y;
				ttime=((coord1>=0.0f) && (coord2>=0.0f) && (t<ttime) && (coord1<=x_size) && (coord2<=y_size))? t : ttime;
				t+=(z_size/ray_dir_z);
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_y_const+t*ray_dir_y;
				ttime=((t<ttime) && (coord1>=0.0f) && (coord2>=0.0f) && (coord1<=x_size) && (coord2<=y_size))? t : ttime;
			}
			if ((ray_dir_y<-0.001f) || (ray_dir_y>0.001f))
			{
				t=-ray_pos_y_const/ray_dir_y;
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((coord1>=0.0f) && (coord2>=0.0f) && (t<ttime) && (coord1<=x_size) && (coord2<=z_size))? t : ttime;
				t+=(y_size/ray_dir_y);
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((t<ttime) && (coord1>=0.0f) && (coord2>=0.0f) && (coord1<=x_size) && (coord2<=z_size))? t : ttime;
			}
			if ((ray_dir_x<-0.001f) || (ray_dir_x>0.001f))
			{
				t=-ray_pos_x_const/ray_dir_x;
				coord1=ray_pos_y_const+t*ray_dir_y;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((coord1>=0.0f) && (coord2>=0.0f) && (t<ttime) && (coord1<=y_size) && (coord2<=z_size))? t : ttime;
				t+=(x_size/ray_dir_x);
				coord1=ray_pos_y_const+t*ray_dir_y;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((t<ttime) && (coord1>=0.0f) && (coord2>=0.0f) && (coord1<=y_size) && (coord2<=z_size))? t : ttime;
			}

			if (ttime>1.0e+29f)
			{
				// no hit - dark gray color
				pixels[i*img_w+j]=0xff969696;
				continue;
			}

			/** traversing the ray in regular grid using a little modified
			    3DDA algorithm (http://ray-tracing.com/articles182.html) */

			/* move ray origin on the bounding box (if this origin is outside the box) */
			ttime=(ttime>0.0f)? ttime : 0.0f;
			ray_pos_x=ray_pos_x_const+ttime*ray_dir_x;
			ray_pos_y=ray_pos_y_const+ttime*ray_dir_y;
			ray_pos_z=ray_pos_z_const+ttime*ray_dir_z;

			if ((ray_pos_x<-0.01f) || (ray_pos_x>x_size+0.01f) || 
				(ray_pos_y<-0.01f) || (ray_pos_y>y_size+0.01f) || 
				(ray_pos_z<-0.01f) || (ray_pos_z>z_size+0.01f))
			{
				// no hit - dark gray color
				pixels[i*img_w+j]=0xff969696;
				continue;
			}

			/* find indexes of the nearest cube to the 'ray_pos' point */
			ind_k=(ray_pos_x<point_step)? 0 : (((ray_pos_x+point_step)>=x_size)? (x_num-1) : (int)floor(ray_pos_x/point_step));
			ind_l=(ray_pos_y<point_step)? 0 : (((ray_pos_y+point_step)>=y_size)? (y_num-1) : (int)floor(ray_pos_y/point_step));
			ind_m=(ray_pos_z<point_step)? 0 : (((ray_pos_z+point_step)>=z_size)? (z_num-1) : (int)floor(ray_pos_z/point_step));

			ind_k_init=ind_k;
			ind_l_init=ind_l;
			ind_m_init=ind_m;

			/* initialize misc */
			r_d_x_ge_0=(int)(ray_dir_x>=0.0f);
			r_d_y_ge_0=(int)(ray_dir_y>=0.0f);
			r_d_z_ge_0=(int)(ray_dir_z>=0.0f);
			ind_k_step=(r_d_x_ge_0<<1)-1;
			ind_l_step=(r_d_y_ge_0<<1)-1;
			ind_m_step=(r_d_z_ge_0<<1)-1;
			t_max_x=(point_step*(float)(ind_k+r_d_x_ge_0)-ray_pos_x_const)/ray_dir_x;
			t_max_y=(point_step*(float)(ind_l+r_d_y_ge_0)-ray_pos_y_const)/ray_dir_y;
			t_max_z=(point_step*(float)(ind_m+r_d_z_ge_0)-ray_pos_z_const)/ray_dir_z;
			t_delta_x=point_step*((float)ind_k_step)/ray_dir_x;
			t_delta_y=point_step*((float)ind_l_step)/ray_dir_y;
			t_delta_z=point_step*((float)ind_m_step)/ray_dir_z;

			xy_num=x_num*y_num*ind_m_step;
			yl_num=x_num*ind_l_step;

			mtr_indx=(ind_m*y_num+ind_l)*x_num+ind_k;

			main_color_g=0.0f;
			main_color_r=0.0f;
			this_point_intense_g=1.0f;
			this_point_intense_r=1.0f;

			/* traverse */
			for ( ; ; )
			{
				/* shading */
				self_intense=((float)GET_G(matrix[mtr_indx]))*this_point_intense_g;
				main_color_g+=self_intense;
				this_point_intense_g-=(self_intense*0.0039216f);
				self_intense=((float)GET_R(matrix[mtr_indx]))*this_point_intense_r;
				main_color_r+=self_intense;
				this_point_intense_r-=(self_intense*0.0039216f);
				if ((main_color_g>253.0f) && (main_color_r>253.0f)) break;

				/* further movement */
				if (t_max_x<=t_max_y)
				{
					if (t_max_x<=t_max_z)
					{
						ind_k+=ind_k_step;
						if ((ind_k>=x_num) || (ind_k<0) || (ind_k_init>ind_k+depth_constraint) || 
							(ind_k_init+depth_constraint<ind_k)) break;
						t_max_x+=t_delta_x;
						mtr_indx+=ind_k_step;
					}
					else
					{
						ind_m+=ind_m_step;
						if ((ind_m>=z_num) || (ind_m<0) || (ind_m_init>ind_m+depth_constraint) || 
							(ind_m_init+depth_constraint<ind_m)) break;
						t_max_z+=t_delta_z;
						mtr_indx+=xy_num;
					}
				}
				else
				{
					if (t_max_y<=t_max_z)
					{
						ind_l+=ind_l_step;
						if ((ind_l>=y_num) || (ind_l<0) || (ind_l_init>ind_l+depth_constraint) || 
							(ind_l_init+depth_constraint<ind_l)) break;
						t_max_y+=t_delta_y;
						mtr_indx+=yl_num;
					}
					else
					{
						ind_m+=ind_m_step;
						if ((ind_m>=z_num) || (ind_m<0) || (ind_m_init>ind_m+depth_constraint) || 
							(ind_m_init+depth_constraint<ind_m)) break;
						t_max_z+=t_delta_z;
						mtr_indx+=xy_num;
					}
				}
			}

			pixels[i*img_w+j]=0xff000000 | 
							((main_color_g>=255.0f)? 0x0000ff00 : (((unsigned int)floor(main_color_g+0.5f))<<8)) | 
							((main_color_r>=255.0f)? 0x00ff0000 : (((unsigned int)floor(main_color_r+0.5f))<<16));
		}
	}
	}
}

void RendererOMP::RenderBox_spheres (unsigned int *pixels) const {
	const float minus_img_w_2=-static_cast<const float>(img_w>>1u);
	const float minus_img_h_2=-static_cast<const float>(img_h>>1u);
	const float ray_dir_z_init=-minus_img_w_2*2.414213562f;
	const float point_radius_sqr=point_radius*point_radius;

	/** all computations will be in bounding box coordinate system! **/

	// "left-bottom-near" point is an origin of bounding box coordinate system
	const float ray_pos_x_const=(cam_x-bbox_cnt_x)*ox_x+(cam_y-bbox_cnt_y)*ox_y+(cam_z-bbox_cnt_z)*ox_z+0.5f*x_size,
	ray_pos_y_const=(cam_x-bbox_cnt_x)*oy_x+(cam_y-bbox_cnt_y)*oy_y+(cam_z-bbox_cnt_z)*oy_z+0.5f*y_size,
	ray_pos_z_const=(cam_x-bbox_cnt_x)*oz_x+(cam_y-bbox_cnt_y)*oz_y+(cam_z-bbox_cnt_z)*oz_z+0.5f*z_size;

	#pragma omp parallel
	{
	unsigned int i,j;
	// Do NOT vectorize all these variables! The speed will be decreased! Oh, you can check...
	float ray_pos_x,ray_pos_y,ray_pos_z,ray_dir_x,ray_dir_y,ray_dir_z;
	float ttime,t;
	float coord1,coord2;
	int ind_k,ind_l,ind_m;
	int ind_k_init,ind_l_init,ind_m_init;
	int ind_k_step,ind_l_step,ind_m_step;
	float t_max_x,t_max_y,t_max_z;
	float t_delta_x,t_delta_y,t_delta_z;
	float to_centre_x,to_centre_y,to_centre_z;
	float ind_step_flt_x,ind_step_flt_y,ind_step_flt_z;
	float main_color_g,main_color_r;
	float self_intense,this_point_intense_g,this_point_intense_r;
	int mtr_indx;
	int xy_num,yl_num;

	#pragma omp for schedule(dynamic,2) nowait
	for (i=0; i<img_h; ++i)
	{
		for (j=0; j<img_w; ++j)
		{
			/* transform ray coordinates */
			ttime=minus_img_w_2+j;
			t=minus_img_h_2+i;
			ray_dir_x=ttime*ox_x+t*ox_y+ray_dir_z_init*ox_z;
			ray_dir_y=ttime*oy_x+t*oy_y+ray_dir_z_init*oy_z;
			ray_dir_z=ttime*oz_x+t*oz_y+ray_dir_z_init*oz_z;
			t=sqrtf(ray_dir_x*ray_dir_x+ray_dir_y*ray_dir_y+ray_dir_z*ray_dir_z);
			ray_dir_x/=t;
			ray_dir_y/=t;
			ray_dir_z/=t;

			ttime=1.0e+30f;

			/* find intersection of the ray and the main bounding box */
			if ((ray_dir_z<-0.001f) || (ray_dir_z>0.001f))
			{
				t=-ray_pos_z_const/ray_dir_z;
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_y_const+t*ray_dir_y;
				ttime=((coord1>=0.0f) && (coord2>=0.0f) && (t<ttime) && (coord1<=x_size) && (coord2<=y_size))? t : ttime;
				t+=(z_size/ray_dir_z);
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_y_const+t*ray_dir_y;
				ttime=((t<ttime) && (coord1>=0.0f) && (coord2>=0.0f) && (coord1<=x_size) && (coord2<=y_size))? t : ttime;
			}
			if ((ray_dir_y<-0.001f) || (ray_dir_y>0.001f))
			{
				t=-ray_pos_y_const/ray_dir_y;
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((coord1>=0.0f) && (coord2>=0.0f) && (t<ttime) && (coord1<=x_size) && (coord2<=z_size))? t : ttime;
				t+=(y_size/ray_dir_y);
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((t<ttime) && (coord1>=0.0f) && (coord2>=0.0f) && (coord1<=x_size) && (coord2<=z_size))? t : ttime;
			}
			if ((ray_dir_x<-0.001f) || (ray_dir_x>0.001f))
			{
				t=-ray_pos_x_const/ray_dir_x;
				coord1=ray_pos_y_const+t*ray_dir_y;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((coord1>=0.0f) && (coord2>=0.0f) && (t<ttime) && (coord1<=y_size) && (coord2<=z_size))? t : ttime;
				t+=(x_size/ray_dir_x);
				coord1=ray_pos_y_const+t*ray_dir_y;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((t<ttime) && (coord1>=0.0f) && (coord2>=0.0f) && (coord1<=y_size) && (coord2<=z_size))? t : ttime;
			}

			if (ttime>1.0e+29f)
			{
				// no hit - dark gray color
				pixels[i*img_w+j]=0xff969696;
				continue;
			}

			/** traversing the ray in regular grid using a little modified
			    3DDA algorithm (http://ray-tracing.com/articles182.html) */

			/* move ray origin on the bounding box (if this origin is outside the box) */
			ttime=(ttime>0.0f)? ttime : 0.0f;
			ray_pos_x=ray_pos_x_const+ttime*ray_dir_x;
			ray_pos_y=ray_pos_y_const+ttime*ray_dir_y;
			ray_pos_z=ray_pos_z_const+ttime*ray_dir_z;

			if ((ray_pos_x<-0.01f) || (ray_pos_x>x_size+0.01f) || 
				(ray_pos_y<-0.01f) || (ray_pos_y>y_size+0.01f) || 
				(ray_pos_z<-0.01f) || (ray_pos_z>z_size+0.01f))
			{
				// no hit - dark gray color
				pixels[i*img_w+j]=0xff969696;
				continue;
			}

			/* find indexes of the nearest cube to the 'ray_pos' point */
			ind_k=(ray_pos_x<point_step)? 0 : (((ray_pos_x+point_step)>=x_size)? (x_num-1) : (int)floor(ray_pos_x/point_step));
			ind_l=(ray_pos_y<point_step)? 0 : (((ray_pos_y+point_step)>=y_size)? (y_num-1) : (int)floor(ray_pos_y/point_step));
			ind_m=(ray_pos_z<point_step)? 0 : (((ray_pos_z+point_step)>=z_size)? (z_num-1) : (int)floor(ray_pos_z/point_step));

			ind_k_init=ind_k;
			ind_l_init=ind_l;
			ind_m_init=ind_m;

			/* initialize misc */
			ind_k_step=((int)(ray_dir_x>=0.0f)<<1)-1;
			ind_l_step=((int)(ray_dir_y>=0.0f)<<1)-1;
			ind_m_step=((int)(ray_dir_z>=0.0f)<<1)-1;
			to_centre_x=point_step*(0.5f+(float)ind_k)-ray_pos_x_const;
			to_centre_y=point_step*(0.5f+(float)ind_l)-ray_pos_y_const;
			to_centre_z=point_step*(0.5f+(float)ind_m)-ray_pos_z_const;
			ind_step_flt_x=point_step*((float)ind_k_step);
			ind_step_flt_y=point_step*((float)ind_l_step);
			ind_step_flt_z=point_step*((float)ind_m_step);
			t_max_x=(to_centre_x+0.5f*ind_step_flt_x)/ray_dir_x;
			t_max_y=(to_centre_y+0.5f*ind_step_flt_y)/ray_dir_y;
			t_max_z=(to_centre_z+0.5f*ind_step_flt_z)/ray_dir_z;
			t_delta_x=ind_step_flt_x/ray_dir_x;
			t_delta_y=ind_step_flt_y/ray_dir_y;
			t_delta_z=ind_step_flt_z/ray_dir_z;

			xy_num=x_num*y_num*ind_m_step;
			yl_num=x_num*ind_l_step;

			mtr_indx=(ind_m*y_num+ind_l)*x_num+ind_k;

			main_color_g=0.0f;
			main_color_r=0.0f;
			this_point_intense_g=1.0f;
			this_point_intense_r=1.0f;

			/* traverse */
			for ( ; ; )
			{
				/* shading */
				t=to_centre_x*ray_dir_x+to_centre_y*ray_dir_y+to_centre_z*ray_dir_z;
				if ((point_radius_sqr+t*t)>(to_centre_x*to_centre_x+to_centre_y*to_centre_y+to_centre_z*to_centre_z))
				{
					self_intense=(float)GET_G(matrix[mtr_indx]);
					main_color_g+=(this_point_intense_g*self_intense);
					this_point_intense_g*=((255.0f-self_intense)*0.0039216f);
					self_intense=(float)GET_R(matrix[mtr_indx]);
					main_color_r+=(this_point_intense_r*self_intense);
					this_point_intense_r*=((255.0f-self_intense)*0.0039216f);
					if ((main_color_g>253.0f) && (main_color_r>253.0f)) break;
				}

				/* further movement */
				if (t_max_x<=t_max_y)
				{
					if (t_max_x<=t_max_z)
					{
						ind_k+=ind_k_step;
						if ((ind_k>=x_num) || (ind_k<0) || (ind_k_init>ind_k+depth_constraint) || 
							(ind_k_init+depth_constraint<ind_k)) break;
						t_max_x+=t_delta_x;
						mtr_indx+=ind_k_step;
						to_centre_x+=ind_step_flt_x;
					}
					else
					{
						ind_m+=ind_m_step;
						if ((ind_m>=z_num) || (ind_m<0) || (ind_m_init>ind_m+depth_constraint) || 
							(ind_m_init+depth_constraint<ind_m)) break;
						t_max_z+=t_delta_z;
						mtr_indx+=xy_num;
						to_centre_z+=ind_step_flt_z;
					}
				}
				else
				{
					if (t_max_y<=t_max_z)
					{
						ind_l+=ind_l_step;
						if ((ind_l>=y_num) || (ind_l<0) || (ind_l_init>ind_l+depth_constraint) || 
							(ind_l_init+depth_constraint<ind_l)) break;
						t_max_y+=t_delta_y;
						mtr_indx+=yl_num;
						to_centre_y+=ind_step_flt_y;
					}
					else
					{
						ind_m+=ind_m_step;
						if ((ind_m>=z_num) || (ind_m<0) || (ind_m_init>ind_m+depth_constraint) || 
							(ind_m_init+depth_constraint<ind_m)) break;
						t_max_z+=t_delta_z;
						mtr_indx+=xy_num;
						to_centre_z+=ind_step_flt_z;
					}
				}
			}

			pixels[i*img_w+j]=0xff000000 | 
							((main_color_g>=255.0f)? 0x0000ff00 : (((unsigned int)floor(main_color_g+0.5f))<<8)) | 
							((main_color_r>=255.0f)? 0x00ff0000 : (((unsigned int)floor(main_color_r+0.5f))<<16));
		}
	}
	}
}

void RendererOMP::RenderBox_lights (unsigned int *pixels) const {
	const float minus_img_w_2=-static_cast<const float>(img_w>>1u);
	const float minus_img_h_2=-static_cast<const float>(img_h>>1u);
	const float ray_dir_z_init=-minus_img_w_2*2.414213562f;
	const float point_radius_sqr=point_radius*point_radius;

	/** all computations will be in bounding box coordinate system! **/

	// "left-bottom-near" point is an origin of bounding box coordinate system
	const float ray_pos_x_const=(cam_x-bbox_cnt_x)*ox_x+(cam_y-bbox_cnt_y)*ox_y+(cam_z-bbox_cnt_z)*ox_z+0.5f*x_size,
	ray_pos_y_const=(cam_x-bbox_cnt_x)*oy_x+(cam_y-bbox_cnt_y)*oy_y+(cam_z-bbox_cnt_z)*oy_z+0.5f*y_size,
	ray_pos_z_const=(cam_x-bbox_cnt_x)*oz_x+(cam_y-bbox_cnt_y)*oz_y+(cam_z-bbox_cnt_z)*oz_z+0.5f*z_size;

	#pragma omp parallel
	{
	unsigned int i,j;
	// Do NOT vectorize all these variables! The speed will be decreased! Oh, you can check...
	float ray_pos_x,ray_pos_y,ray_pos_z,ray_dir_x,ray_dir_y,ray_dir_z;
	float ttime,t;
	float coord1,coord2;
	int ind_k,ind_l,ind_m;
	int ind_k_init,ind_l_init,ind_m_init;
	int ind_k_step,ind_l_step,ind_m_step;
	float t_max_x,t_max_y,t_max_z;
	float t_delta_x,t_delta_y,t_delta_z;
	float to_centre_x,to_centre_y,to_centre_z;
	float ind_step_flt_x,ind_step_flt_y,ind_step_flt_z;
	float main_color_g,main_color_r;
	float self_intense,this_point_intense_g,this_point_intense_r;
	int mtr_indx;
	int xy_num,yl_num;

	#pragma omp for schedule(dynamic,2) nowait
	for (i=0; i<img_h; ++i)
	{
		for (j=0; j<img_w; ++j)
		{
			/* transform ray coordinates */
			ttime=minus_img_w_2+j;
			t=minus_img_h_2+i;
			ray_dir_x=ttime*ox_x+t*ox_y+ray_dir_z_init*ox_z;
			ray_dir_y=ttime*oy_x+t*oy_y+ray_dir_z_init*oy_z;
			ray_dir_z=ttime*oz_x+t*oz_y+ray_dir_z_init*oz_z;
			t=sqrtf(ray_dir_x*ray_dir_x+ray_dir_y*ray_dir_y+ray_dir_z*ray_dir_z);
			ray_dir_x/=t;
			ray_dir_y/=t;
			ray_dir_z/=t;

			ttime=1.0e+30f;

			/* find intersection of the ray and the main bounding box */
			if ((ray_dir_z<-0.001f) || (ray_dir_z>0.001f))
			{
				t=-ray_pos_z_const/ray_dir_z;
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_y_const+t*ray_dir_y;
				ttime=((coord1>=0.0f) && (coord2>=0.0f) && (t<ttime) && (coord1<=x_size) && (coord2<=y_size))? t : ttime;
				t+=(z_size/ray_dir_z);
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_y_const+t*ray_dir_y;
				ttime=((t<ttime) && (coord1>=0.0f) && (coord2>=0.0f) && (coord1<=x_size) && (coord2<=y_size))? t : ttime;
			}
			if ((ray_dir_y<-0.001f) || (ray_dir_y>0.001f))
			{
				t=-ray_pos_y_const/ray_dir_y;
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((coord1>=0.0f) && (coord2>=0.0f) && (t<ttime) && (coord1<=x_size) && (coord2<=z_size))? t : ttime;
				t+=(y_size/ray_dir_y);
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((t<ttime) && (coord1>=0.0f) && (coord2>=0.0f) && (coord1<=x_size) && (coord2<=z_size))? t : ttime;
			}
			if ((ray_dir_x<-0.001f) || (ray_dir_x>0.001f))
			{
				t=-ray_pos_x_const/ray_dir_x;
				coord1=ray_pos_y_const+t*ray_dir_y;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((coord1>=0.0f) && (coord2>=0.0f) && (t<ttime) && (coord1<=y_size) && (coord2<=z_size))? t : ttime;
				t+=(x_size/ray_dir_x);
				coord1=ray_pos_y_const+t*ray_dir_y;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((t<ttime) && (coord1>=0.0f) && (coord2>=0.0f) && (coord1<=y_size) && (coord2<=z_size))? t : ttime;
			}

			if (ttime>1.0e+29f)
			{
				// no hit - dark gray color
				pixels[i*img_w+j]=0xff969696;
				continue;
			}

			/** traversing the ray in regular grid using a little modified
			    3DDA algorithm (http://ray-tracing.com/articles182.html) */

			/* move ray origin on the bounding box (if this origin is outside the box) */
			ttime=(ttime>0.0f)? ttime : 0.0f;
			ray_pos_x=ray_pos_x_const+ttime*ray_dir_x;
			ray_pos_y=ray_pos_y_const+ttime*ray_dir_y;
			ray_pos_z=ray_pos_z_const+ttime*ray_dir_z;

			if ((ray_pos_x<-0.01f) || (ray_pos_x>x_size+0.01f) || 
				(ray_pos_y<-0.01f) || (ray_pos_y>y_size+0.01f) || 
				(ray_pos_z<-0.01f) || (ray_pos_z>z_size+0.01f))
			{
				// no hit - dark gray color
				pixels[i*img_w+j]=0xff969696;
				continue;
			}

			/* find indexes of the nearest cube to the 'ray_pos' point */
			ind_k=(ray_pos_x<point_step)? 0 : (((ray_pos_x+point_step)>=x_size)? (x_num-1) : (int)floor(ray_pos_x/point_step));
			ind_l=(ray_pos_y<point_step)? 0 : (((ray_pos_y+point_step)>=y_size)? (y_num-1) : (int)floor(ray_pos_y/point_step));
			ind_m=(ray_pos_z<point_step)? 0 : (((ray_pos_z+point_step)>=z_size)? (z_num-1) : (int)floor(ray_pos_z/point_step));

			ind_k_init=ind_k;
			ind_l_init=ind_l;
			ind_m_init=ind_m;

			/* initialize misc */
			ind_k_step=((int)(ray_dir_x>=0.0f)<<1)-1;
			ind_l_step=((int)(ray_dir_y>=0.0f)<<1)-1;
			ind_m_step=((int)(ray_dir_z>=0.0f)<<1)-1;
			to_centre_x=point_step*(0.5f+(float)ind_k)-ray_pos_x_const;
			to_centre_y=point_step*(0.5f+(float)ind_l)-ray_pos_y_const;
			to_centre_z=point_step*(0.5f+(float)ind_m)-ray_pos_z_const;
			ind_step_flt_x=point_step*static_cast<float>(ind_k_step);
			ind_step_flt_y=point_step*static_cast<float>(ind_l_step);
			ind_step_flt_z=point_step*static_cast<float>(ind_m_step);
			t_max_x=(to_centre_x+0.5f*ind_step_flt_x)/ray_dir_x;
			t_max_y=(to_centre_y+0.5f*ind_step_flt_y)/ray_dir_y;
			t_max_z=(to_centre_z+0.5f*ind_step_flt_z)/ray_dir_z;
			t_delta_x=ind_step_flt_x/ray_dir_x;
			t_delta_y=ind_step_flt_y/ray_dir_y;
			t_delta_z=ind_step_flt_z/ray_dir_z;

			xy_num=x_num*y_num*ind_m_step;
			yl_num=x_num*ind_l_step;

			mtr_indx=(ind_m*y_num+ind_l)*x_num+ind_k;

			main_color_g=0.0f;
			main_color_r=0.0f;
			this_point_intense_g=1.0f;
			this_point_intense_r=1.0f;

			/* traverse */
			for ( ; ; )
			{
				/* shading */
				ttime=to_centre_x*ray_dir_x+to_centre_y*ray_dir_y+to_centre_z*ray_dir_z;
				t=to_centre_x*to_centre_x+to_centre_y*to_centre_y+(to_centre_z-ttime)*(to_centre_z+ttime);
				if (t<point_radius_sqr)
				{
					t=(point_radius_sqr-t)/point_radius_sqr; // fading of intensity (~1/r^2)
					self_intense=((float)GET_G(matrix[mtr_indx]))*t;
					main_color_g+=(this_point_intense_g*self_intense);
					this_point_intense_g*=((255.0f-self_intense)*0.0039216f);
					self_intense=((float)GET_R(matrix[mtr_indx]))*t;
					main_color_r+=(this_point_intense_r*self_intense);
					this_point_intense_r*=((255.0f-self_intense)*0.0039216f);
					if ((main_color_g>253.0f) && (main_color_r>253.0f)) break;
				}

				/* further movement */
				if (t_max_x<=t_max_y)
				{
					if (t_max_x<=t_max_z)
					{
						ind_k+=ind_k_step;
						if ((ind_k>=x_num) || (ind_k<0) || (ind_k_init>ind_k+depth_constraint) || 
							(ind_k_init+depth_constraint<ind_k)) break;
						t_max_x+=t_delta_x;
						mtr_indx+=ind_k_step;
						to_centre_x+=ind_step_flt_x;
					}
					else
					{
						ind_m+=ind_m_step;
						if ((ind_m>=z_num) || (ind_m<0) || (ind_m_init>ind_m+depth_constraint) || 
							(ind_m_init+depth_constraint<ind_m)) break;
						t_max_z+=t_delta_z;
						mtr_indx+=xy_num;
						to_centre_z+=ind_step_flt_z;
					}
				}
				else
				{
					if (t_max_y<=t_max_z)
					{
						ind_l+=ind_l_step;
						if ((ind_l>=y_num) || (ind_l<0) || (ind_l_init>ind_l+depth_constraint) || 
							(ind_l_init+depth_constraint<ind_l)) break;
						t_max_y+=t_delta_y;
						mtr_indx+=yl_num;
						to_centre_y+=ind_step_flt_y;
					}
					else
					{
						ind_m+=ind_m_step;
						if ((ind_m>=z_num) || (ind_m<0) || (ind_m_init>ind_m+depth_constraint) || 
							(ind_m_init+depth_constraint<ind_m)) break;
						t_max_z+=t_delta_z;
						mtr_indx+=xy_num;
						to_centre_z+=ind_step_flt_z;
					}
				}
			}

			pixels[i*img_w+j]=0xff000000 | 
							((main_color_g<255.0f)? (((unsigned int)floor(main_color_g+0.5f))<<8u) : 0x0000ff00) | 
							((main_color_r<255.0f)? (((unsigned int)floor(main_color_r+0.5f))<<16u) : 0x00ff0000);
		}
	}
	}
}

void RendererOMP::RenderBox_cubes_vol (unsigned int *pixels) const {
	const float minus_img_w_2=-static_cast<const float>(img_w>>1u);
	const float minus_img_h_2=-static_cast<const float>(img_h>>1u);
	const float ray_dir_z_init=-minus_img_w_2*2.414213562f;

	/** all computations will be in bounding box coordinate system! **/

	// "left-bottom-near" point is an origin of bounding box coordinate system
	const float ray_pos_x_const=(cam_x-bbox_cnt_x)*ox_x+(cam_y-bbox_cnt_y)*ox_y+(cam_z-bbox_cnt_z)*ox_z+0.5f*x_size,
	ray_pos_y_const=(cam_x-bbox_cnt_x)*oy_x+(cam_y-bbox_cnt_y)*oy_y+(cam_z-bbox_cnt_z)*oy_z+0.5f*y_size,
	ray_pos_z_const=(cam_x-bbox_cnt_x)*oz_x+(cam_y-bbox_cnt_y)*oz_y+(cam_z-bbox_cnt_z)*oz_z+0.5f*z_size;
	// color constraints
	const unsigned char min_green=(unsigned char)(clr_minmax & 0xff);
	const unsigned char max_green=(unsigned char)((clr_minmax>>8u) & 0xff);
	const unsigned char min_red=(unsigned char)((clr_minmax>>16u) & 0xff);
	const unsigned char max_red=(unsigned char)((clr_minmax>>24u) & 0xff);

	#pragma omp parallel
	{
	unsigned int i,j;
	// Do NOT vectorize these variables! The speed will be decreased! Oh, you can check...
	float ray_pos_x,ray_pos_y,ray_pos_z,ray_dir_x,ray_dir_y,ray_dir_z;
	float ttime,t;
	float coord1,coord2;
	int ind_k,ind_l,ind_m;
	int ind_k_step,ind_l_step,ind_m_step;
	float t_max_x,t_max_y,t_max_z;
	float t_delta_x,t_delta_y,t_delta_z;
	float main_color_g,main_color_r;
	float self_intense,this_point_intense_g,this_point_intense_r;
	int r_d_x_ge_0,r_d_y_ge_0,r_d_z_ge_0;
	int mtr_indx;
	int xy_num,yl_num;
	unsigned char color_g,color_r;

	#pragma omp for schedule(dynamic,2) nowait
	for (i=0; i<img_h; ++i)
	{
		for (j=0; j<img_w; ++j)
		{
			/* transform ray coordinates */
			ttime=minus_img_w_2+j;
			t=minus_img_h_2+i;
			ray_dir_x=ttime*ox_x+t*ox_y+ray_dir_z_init*ox_z;
			ray_dir_y=ttime*oy_x+t*oy_y+ray_dir_z_init*oy_z;
			ray_dir_z=ttime*oz_x+t*oz_y+ray_dir_z_init*oz_z;
			t=sqrtf(ray_dir_x*ray_dir_x+ray_dir_y*ray_dir_y+ray_dir_z*ray_dir_z);
			ray_dir_x/=t;
			ray_dir_y/=t;
			ray_dir_z/=t;

			ttime=1.0e+30f;

			/* find intersection of the ray and the main bounding box */
			if ((ray_dir_z<-0.001f) || (ray_dir_z>0.001f))
			{
				t=-ray_pos_z_const/ray_dir_z;
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_y_const+t*ray_dir_y;
				ttime=((coord1>=0.0f) && (coord2>=0.0f) && (t<ttime) && (coord1<=x_size) && (coord2<=y_size))? t : ttime;
				t+=(z_size/ray_dir_z);
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_y_const+t*ray_dir_y;
				ttime=((t<ttime) && (coord1>=0.0f) && (coord2>=0.0f) && (coord1<=x_size) && (coord2<=y_size))? t : ttime;
			}
			if ((ray_dir_y<-0.001f) || (ray_dir_y>0.001f))
			{
				t=-ray_pos_y_const/ray_dir_y;
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((coord1>=0.0f) && (coord2>=0.0f) && (t<ttime) && (coord1<=x_size) && (coord2<=z_size))? t : ttime;
				t+=(y_size/ray_dir_y);
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((t<ttime) && (coord1>=0.0f) && (coord2>=0.0f) && (coord1<=x_size) && (coord2<=z_size))? t : ttime;
			}
			if ((ray_dir_x<-0.001f) || (ray_dir_x>0.001f))
			{
				t=-ray_pos_x_const/ray_dir_x;
				coord1=ray_pos_y_const+t*ray_dir_y;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((coord1>=0.0f) && (coord2>=0.0f) && (t<ttime) && (coord1<=y_size) && (coord2<=z_size))? t : ttime;
				t+=(x_size/ray_dir_x);
				coord1=ray_pos_y_const+t*ray_dir_y;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((t<ttime) && (coord1>=0.0f) && (coord2>=0.0f) && (coord1<=y_size) && (coord2<=z_size))? t : ttime;
			}

			if (ttime>1.0e+29f)
			{
				// no hit - dark gray color
				pixels[i*img_w+j]=0xff969696;
				continue;
			}

			/** traversing the ray in regular grid using a little modified
			    3DDA algorithm (http://ray-tracing.com/articles182.html) */

			/* move ray origin on the bounding box (if this origin is outside the box) */
			ttime=(ttime>0.0f)? ttime : 0.0f;
			ray_pos_x=ray_pos_x_const+ttime*ray_dir_x;
			ray_pos_y=ray_pos_y_const+ttime*ray_dir_y;
			ray_pos_z=ray_pos_z_const+ttime*ray_dir_z;

			if ((ray_pos_x<-0.01f) || (ray_pos_x>x_size+0.01f) || 
				(ray_pos_y<-0.01f) || (ray_pos_y>y_size+0.01f) || 
				(ray_pos_z<-0.01f) || (ray_pos_z>z_size+0.01f))
			{
				// no hit - dark gray color
				pixels[i*img_w+j]=0xff969696;
				continue;
			}

			/* find indexes of the nearest cube to the 'ray_pos' point */
			ind_k=(ray_pos_x<point_step)? 0 : (((ray_pos_x+point_step)>=x_size)? (x_num-1) : (int)floor(ray_pos_x/point_step));
			ind_l=(ray_pos_y<point_step)? 0 : (((ray_pos_y+point_step)>=y_size)? (y_num-1) : (int)floor(ray_pos_y/point_step));
			ind_m=(ray_pos_z<point_step)? 0 : (((ray_pos_z+point_step)>=z_size)? (z_num-1) : (int)floor(ray_pos_z/point_step));

			/* initialize misc */
			r_d_x_ge_0=(int)(ray_dir_x>=0.0f);
			r_d_y_ge_0=(int)(ray_dir_y>=0.0f);
			r_d_z_ge_0=(int)(ray_dir_z>=0.0f);
			ind_k_step=(r_d_x_ge_0<<1)-1;
			ind_l_step=(r_d_y_ge_0<<1)-1;
			ind_m_step=(r_d_z_ge_0<<1)-1;
			t_max_x=(point_step*(float)(ind_k+r_d_x_ge_0)-ray_pos_x_const)/ray_dir_x;
			t_max_y=(point_step*(float)(ind_l+r_d_y_ge_0)-ray_pos_y_const)/ray_dir_y;
			t_max_z=(point_step*(float)(ind_m+r_d_z_ge_0)-ray_pos_z_const)/ray_dir_z;
			t_delta_x=point_step*((float)ind_k_step)/ray_dir_x;
			t_delta_y=point_step*((float)ind_l_step)/ray_dir_y;
			t_delta_z=point_step*((float)ind_m_step)/ray_dir_z;

			xy_num=x_num*y_num*ind_m_step;
			yl_num=x_num*ind_l_step;

			mtr_indx=(ind_m*y_num+ind_l)*x_num+ind_k;

			main_color_g=0.0f;
			main_color_r=0.0f;
			this_point_intense_g=1.0f;
			this_point_intense_r=1.0f;

			/* traverse */
			for ( ; ; )
			{
				/* shading */
				color_g=GET_G(matrix[mtr_indx]);
				color_r=GET_R(matrix[mtr_indx]);
				if ((color_g>=min_green) && (color_g<=max_green) && (color_r>=min_red) && (color_r<=max_red))
				{
					self_intense=(float)color_g;
					main_color_g+=(this_point_intense_g*self_intense);
					this_point_intense_g*=((255.0f-self_intense)*0.0039216f);
					self_intense=(float)color_r;
					main_color_r+=(this_point_intense_r*self_intense);
					this_point_intense_r*=((255.0f-self_intense)*0.0039216f);
					if ((main_color_g>253.0f) && (main_color_r>253.0f)) break;
				}

				/* further movement */
				if (t_max_x<=t_max_y)
				{
					if (t_max_x<=t_max_z)
					{
						ind_k+=ind_k_step;
						if ((ind_k>=x_num) || (ind_k<0)) break;
						t_max_x+=t_delta_x;
						mtr_indx+=ind_k_step;
					}
					else
					{
						ind_m+=ind_m_step;
						if ((ind_m>=z_num) || (ind_m<0)) break;
						t_max_z+=t_delta_z;
						mtr_indx+=xy_num;
					}
				}
				else
				{
					if (t_max_y<=t_max_z)
					{
						ind_l+=ind_l_step;
						if ((ind_l>=y_num) || (ind_l<0)) break;
						t_max_y+=t_delta_y;
						mtr_indx+=yl_num;
					}
					else
					{
						ind_m+=ind_m_step;
						if ((ind_m>=z_num) || (ind_m<0)) break;
						t_max_z+=t_delta_z;
						mtr_indx+=xy_num;
					}
				}
			}

			if ((main_color_g==0.0f) && (main_color_r==0.0f))
				pixels[i*img_w+j]=0xff969696;
			else
				pixels[i*img_w+j]=0xff000000 | 
							((main_color_g>=255.0f)? 0x0000ff00 : (((unsigned int)floor(main_color_g+0.5f))<<8)) | 
							((main_color_r>=255.0f)? 0x00ff0000 : (((unsigned int)floor(main_color_r+0.5f))<<16));
		}
	}
	}
}

void RendererOMP::RenderBox_spheres_vol (unsigned int *pixels) const {
	const float minus_img_w_2=-static_cast<const float>(img_w>>1u);
	const float minus_img_h_2=-static_cast<const float>(img_h>>1u);
	const float ray_dir_z_init=-minus_img_w_2*2.414213562f;
	const float point_radius_sqr=point_radius*point_radius;

	/** all computations will be in bounding box coordinate system! **/

	// "left-bottom-near" point is an origin of bounding box coordinate system
	const float ray_pos_x_const=(cam_x-bbox_cnt_x)*ox_x+(cam_y-bbox_cnt_y)*ox_y+(cam_z-bbox_cnt_z)*ox_z+0.5f*x_size,
	ray_pos_y_const=(cam_x-bbox_cnt_x)*oy_x+(cam_y-bbox_cnt_y)*oy_y+(cam_z-bbox_cnt_z)*oy_z+0.5f*y_size,
	ray_pos_z_const=(cam_x-bbox_cnt_x)*oz_x+(cam_y-bbox_cnt_y)*oz_y+(cam_z-bbox_cnt_z)*oz_z+0.5f*z_size;
	// color constraints
	const unsigned char min_green=(unsigned char)(clr_minmax & 0xff);
	const unsigned char max_green=(unsigned char)((clr_minmax>>8) & 0xff);
	const unsigned char min_red=(unsigned char)((clr_minmax>>16) & 0xff);
	const unsigned char max_red=(unsigned char)((clr_minmax>>24) & 0xff);

	#pragma omp parallel
	{
	unsigned int i,j;
	// Do NOT vectorize all these variables! The speed will be decreased! Oh, you can check...
	float ray_pos_x,ray_pos_y,ray_pos_z,ray_dir_x,ray_dir_y,ray_dir_z;
	float ttime,t;
	float coord1,coord2;
	int ind_k,ind_l,ind_m;
	int ind_k_step,ind_l_step,ind_m_step;
	float t_max_x,t_max_y,t_max_z;
	float t_delta_x,t_delta_y,t_delta_z;
	float to_centre_x,to_centre_y,to_centre_z;
	float ind_step_flt_x,ind_step_flt_y,ind_step_flt_z;
	float main_color_g,main_color_r;
	float self_intense,this_point_intense_g,this_point_intense_r;
	int mtr_indx;
	int xy_num,yl_num;
	unsigned char color_g,color_r;

	#pragma omp for schedule(dynamic,2) nowait
	for (i=0; i<img_h; ++i)
	{
		for (j=0; j<img_w; ++j)
		{
			/* transform ray coordinates */
			ttime=minus_img_w_2+j;
			t=minus_img_h_2+i;
			ray_dir_x=ttime*ox_x+t*ox_y+ray_dir_z_init*ox_z;
			ray_dir_y=ttime*oy_x+t*oy_y+ray_dir_z_init*oy_z;
			ray_dir_z=ttime*oz_x+t*oz_y+ray_dir_z_init*oz_z;
			t=sqrtf(ray_dir_x*ray_dir_x+ray_dir_y*ray_dir_y+ray_dir_z*ray_dir_z);
			ray_dir_x/=t;
			ray_dir_y/=t;
			ray_dir_z/=t;

			ttime=1.0e+30f;

			/* find intersection of the ray and the main bounding box */
			if ((ray_dir_z<-0.001f) || (ray_dir_z>0.001f))
			{
				t=-ray_pos_z_const/ray_dir_z;
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_y_const+t*ray_dir_y;
				ttime=((coord1>=0.0f) && (coord2>=0.0f) && (t<ttime) && (coord1<=x_size) && (coord2<=y_size))? t : ttime;
				t+=(z_size/ray_dir_z);
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_y_const+t*ray_dir_y;
				ttime=((t<ttime) && (coord1>=0.0f) && (coord2>=0.0f) && (coord1<=x_size) && (coord2<=y_size))? t : ttime;
			}
			if ((ray_dir_y<-0.001f) || (ray_dir_y>0.001f))
			{
				t=-ray_pos_y_const/ray_dir_y;
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((coord1>=0.0f) && (coord2>=0.0f) && (t<ttime) && (coord1<=x_size) && (coord2<=z_size))? t : ttime;
				t+=(y_size/ray_dir_y);
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((t<ttime) && (coord1>=0.0f) && (coord2>=0.0f) && (coord1<=x_size) && (coord2<=z_size))? t : ttime;
			}
			if ((ray_dir_x<-0.001f) || (ray_dir_x>0.001f))
			{
				t=-ray_pos_x_const/ray_dir_x;
				coord1=ray_pos_y_const+t*ray_dir_y;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((coord1>=0.0f) && (coord2>=0.0f) && (t<ttime) && (coord1<=y_size) && (coord2<=z_size))? t : ttime;
				t+=(x_size/ray_dir_x);
				coord1=ray_pos_y_const+t*ray_dir_y;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((t<ttime) && (coord1>=0.0f) && (coord2>=0.0f) && (coord1<=y_size) && (coord2<=z_size))? t : ttime;
			}

			if (ttime>1.0e+29f)
			{
				// no hit - dark gray color
				pixels[i*img_w+j]=0xff969696;
				continue;
			}

			/** traversing the ray in regular grid using a little modified
			    3DDA algorithm (http://ray-tracing.com/articles182.html) */

			/* move ray origin on the bounding box (if this origin is outside the box) */
			ttime=(ttime>0.0f)? ttime : 0.0f;
			ray_pos_x=ray_pos_x_const+ttime*ray_dir_x;
			ray_pos_y=ray_pos_y_const+ttime*ray_dir_y;
			ray_pos_z=ray_pos_z_const+ttime*ray_dir_z;

			if ((ray_pos_x<-0.01f) || (ray_pos_x>x_size+0.01f) || 
				(ray_pos_y<-0.01f) || (ray_pos_y>y_size+0.01f) || 
				(ray_pos_z<-0.01f) || (ray_pos_z>z_size+0.01f))
			{
				// no hit - dark gray color
				pixels[i*img_w+j]=0xff969696;
				continue;
			}

			/* find indexes of the nearest cube to the 'ray_pos' point */
			ind_k=(ray_pos_x<point_step)? 0 : (((ray_pos_x+point_step)>=x_size)? (x_num-1) : (int)floor(ray_pos_x/point_step));
			ind_l=(ray_pos_y<point_step)? 0 : (((ray_pos_y+point_step)>=y_size)? (y_num-1) : (int)floor(ray_pos_y/point_step));
			ind_m=(ray_pos_z<point_step)? 0 : (((ray_pos_z+point_step)>=z_size)? (z_num-1) : (int)floor(ray_pos_z/point_step));

			/* initialize misc */
			ind_k_step=((int)(ray_dir_x>=0.0f)<<1)-1;
			ind_l_step=((int)(ray_dir_y>=0.0f)<<1)-1;
			ind_m_step=((int)(ray_dir_z>=0.0f)<<1)-1;
			to_centre_x=point_step*(0.5f+(float)ind_k)-ray_pos_x_const;
			to_centre_y=point_step*(0.5f+(float)ind_l)-ray_pos_y_const;
			to_centre_z=point_step*(0.5f+(float)ind_m)-ray_pos_z_const;
			ind_step_flt_x=point_step*((float)ind_k_step);
			ind_step_flt_y=point_step*((float)ind_l_step);
			ind_step_flt_z=point_step*((float)ind_m_step);
			t_max_x=(to_centre_x+0.5f*ind_step_flt_x)/ray_dir_x;
			t_max_y=(to_centre_y+0.5f*ind_step_flt_y)/ray_dir_y;
			t_max_z=(to_centre_z+0.5f*ind_step_flt_z)/ray_dir_z;
			t_delta_x=ind_step_flt_x/ray_dir_x;
			t_delta_y=ind_step_flt_y/ray_dir_y;
			t_delta_z=ind_step_flt_z/ray_dir_z;

			xy_num=x_num*y_num*ind_m_step;
			yl_num=x_num*ind_l_step;

			mtr_indx=(ind_m*y_num+ind_l)*x_num+ind_k;

			main_color_g=0.0f;
			main_color_r=0.0f;
			this_point_intense_g=1.0f;
			this_point_intense_r=1.0f;

			/* traverse */
			for ( ; ; )
			{
				/* shading */
				t=to_centre_x*ray_dir_x+to_centre_y*ray_dir_y+to_centre_z*ray_dir_z;
				if ((point_radius_sqr+t*t)>(to_centre_x*to_centre_x+to_centre_y*to_centre_y+to_centre_z*to_centre_z))
				{
					color_g=GET_G(matrix[mtr_indx]);
					color_r=GET_R(matrix[mtr_indx]);
					if ((color_g>=min_green) && (color_g<=max_green) && (color_r>=min_red) && (color_r<=max_red))
					{
						self_intense=(float)color_g;
						main_color_g+=(this_point_intense_g*self_intense);
						this_point_intense_g*=((255.0f-self_intense)*0.0039216f);
						self_intense=(float)color_r;
						main_color_r+=(this_point_intense_r*self_intense);
						this_point_intense_r*=((255.0f-self_intense)*0.0039216f);
						if ((main_color_g>253.0f) && (main_color_r>253.0f)) break;
					}
				}

				/* further movement */
				if (t_max_x<=t_max_y)
				{
					if (t_max_x<=t_max_z)
					{
						ind_k+=ind_k_step;
						if ((ind_k>=x_num) || (ind_k<0)) break;
						t_max_x+=t_delta_x;
						mtr_indx+=ind_k_step;
						to_centre_x+=ind_step_flt_x;
					}
					else
					{
						ind_m+=ind_m_step;
						if ((ind_m>=z_num) || (ind_m<0)) break;
						t_max_z+=t_delta_z;
						mtr_indx+=xy_num;
						to_centre_z+=ind_step_flt_z;
					}
				}
				else
				{
					if (t_max_y<=t_max_z)
					{
						ind_l+=ind_l_step;
						if ((ind_l>=y_num) || (ind_l<0)) break;
						t_max_y+=t_delta_y;
						mtr_indx+=yl_num;
						to_centre_y+=ind_step_flt_y;
					}
					else
					{
						ind_m+=ind_m_step;
						if ((ind_m>=z_num) || (ind_m<0)) break;
						t_max_z+=t_delta_z;
						mtr_indx+=xy_num;
						to_centre_z+=ind_step_flt_z;
					}
				}
			}

			if ((main_color_g==0.0f) && (main_color_r==0.0f))
				pixels[i*img_w+j]=0xff969696;
			else
				pixels[i*img_w+j]=0xff000000 | 
							((main_color_g>=255.0f)? 0x0000ff00 : (((unsigned int)floor(main_color_g+0.5f))<<8)) | 
							((main_color_r>=255.0f)? 0x00ff0000 : (((unsigned int)floor(main_color_r+0.5f))<<16));
		}
	}
	}
}

void RendererOMP::RenderBox_lights_vol (unsigned int *pixels) const {
	const float minus_img_w_2=-static_cast<const float>(img_w>>1u);
	const float minus_img_h_2=-static_cast<const float>(img_h>>1u);
	const float ray_dir_z_init=-minus_img_w_2*2.414213562f;
	const float point_radius_sqr=point_radius*point_radius;

	/** all computations will be in bounding box coordinate system! **/

	// "left-bottom-near" point is an origin of bounding box coordinate system
	const float ray_pos_x_const=(cam_x-bbox_cnt_x)*ox_x+(cam_y-bbox_cnt_y)*ox_y+(cam_z-bbox_cnt_z)*ox_z+0.5f*x_size,
	ray_pos_y_const=(cam_x-bbox_cnt_x)*oy_x+(cam_y-bbox_cnt_y)*oy_y+(cam_z-bbox_cnt_z)*oy_z+0.5f*y_size,
	ray_pos_z_const=(cam_x-bbox_cnt_x)*oz_x+(cam_y-bbox_cnt_y)*oz_y+(cam_z-bbox_cnt_z)*oz_z+0.5f*z_size;
	// color constraints
	const unsigned char min_green=(unsigned char)(clr_minmax & 0xff);
	const unsigned char max_green=(unsigned char)((clr_minmax>>8) & 0xff);
	const unsigned char min_red=(unsigned char)((clr_minmax>>16) & 0xff);
	const unsigned char max_red=(unsigned char)((clr_minmax>>24) & 0xff);

	#pragma omp parallel
	{
	unsigned int i,j;
	// Do NOT vectorize all these variables! The speed will be decreased! Oh, you can check...
	float ray_pos_x,ray_pos_y,ray_pos_z,ray_dir_x,ray_dir_y,ray_dir_z;
	float ttime,t;
	float coord1,coord2;
	int ind_k,ind_l,ind_m;
	int ind_k_step,ind_l_step,ind_m_step;
	float t_max_x,t_max_y,t_max_z;
	float t_delta_x,t_delta_y,t_delta_z;
	float to_centre_x,to_centre_y,to_centre_z;
	float ind_step_flt_x,ind_step_flt_y,ind_step_flt_z;
	float main_color_g,main_color_r;
	float self_intense,this_point_intense_g,this_point_intense_r;
	int mtr_indx;
	int xy_num,yl_num;
	unsigned char color_g,color_r;

	#pragma omp for schedule(dynamic,2) nowait
	for (i=0; i<img_h; ++i)
	{
		for (j=0; j<img_w; ++j)
		{	    
			/* transform ray coordinates */
			ttime=minus_img_w_2+j;
			t=minus_img_h_2+i;
			ray_dir_x=ttime*ox_x+t*ox_y+ray_dir_z_init*ox_z;
			ray_dir_y=ttime*oy_x+t*oy_y+ray_dir_z_init*oy_z;
			ray_dir_z=ttime*oz_x+t*oz_y+ray_dir_z_init*oz_z;
			t=sqrtf(ray_dir_x*ray_dir_x+ray_dir_y*ray_dir_y+ray_dir_z*ray_dir_z);
			ray_dir_x/=t;
			ray_dir_y/=t;
			ray_dir_z/=t;

			ttime=1.0e+30f;

			/* find intersection of the ray and the main bounding box */
			if ((ray_dir_z<-0.001f) || (ray_dir_z>0.001f))
			{
				t=-ray_pos_z_const/ray_dir_z;
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_y_const+t*ray_dir_y;
				ttime=((coord1>=0.0f) && (coord2>=0.0f) && (t<ttime) && (coord1<=x_size) && (coord2<=y_size))? t : ttime;
				t+=(z_size/ray_dir_z);
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_y_const+t*ray_dir_y;
				ttime=((t<ttime) && (coord1>=0.0f) && (coord2>=0.0f) && (coord1<=x_size) && (coord2<=y_size))? t : ttime;
			}
			if ((ray_dir_y<-0.001f) || (ray_dir_y>0.001f))
			{
				t=-ray_pos_y_const/ray_dir_y;
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((coord1>=0.0f) && (coord2>=0.0f) && (t<ttime) && (coord1<=x_size) && (coord2<=z_size))? t : ttime;
				t+=(y_size/ray_dir_y);
				coord1=ray_pos_x_const+t*ray_dir_x;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((t<ttime) && (coord1>=0.0f) && (coord2>=0.0f) && (coord1<=x_size) && (coord2<=z_size))? t : ttime;
			}
			if ((ray_dir_x<-0.001f) || (ray_dir_x>0.001f))
			{
				t=-ray_pos_x_const/ray_dir_x;
				coord1=ray_pos_y_const+t*ray_dir_y;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((coord1>=0.0f) && (coord2>=0.0f) && (t<ttime) && (coord1<=y_size) && (coord2<=z_size))? t : ttime;
				t+=(x_size/ray_dir_x);
				coord1=ray_pos_y_const+t*ray_dir_y;
				coord2=ray_pos_z_const+t*ray_dir_z;
				ttime=((t<ttime) && (coord1>=0.0f) && (coord2>=0.0f) && (coord1<=y_size) && (coord2<=z_size))? t : ttime;
			}

			if (ttime>1.0e+29f)
			{
				// no hit - dark gray color
				pixels[i*img_w+j]=0xff969696;
				continue;
			}

			/** traversing the ray in regular grid using a little modified
			    3DDA algorithm (http://ray-tracing.com/articles182.html) */

			/* move ray origin on the bounding box (if this origin is outside the box) */
			ttime=(ttime>0.0f)? ttime : 0.0f;
			ray_pos_x=ray_pos_x_const+ttime*ray_dir_x;
			ray_pos_y=ray_pos_y_const+ttime*ray_dir_y;
			ray_pos_z=ray_pos_z_const+ttime*ray_dir_z;

			if ((ray_pos_x<-0.01f) || (ray_pos_x>x_size+0.01f) || 
				(ray_pos_y<-0.01f) || (ray_pos_y>y_size+0.01f) || 
				(ray_pos_z<-0.01f) || (ray_pos_z>z_size+0.01f))
			{
				// no hit - dark gray color
				pixels[i*img_w+j]=0xff969696;
				continue;
			}

			/* find indexes of the nearest cube to the 'ray_pos' point */
			ind_k=(ray_pos_x<point_step)? 0 : (((ray_pos_x+point_step)>=x_size)? (x_num-1) : (int)floor(ray_pos_x/point_step));
			ind_l=(ray_pos_y<point_step)? 0 : (((ray_pos_y+point_step)>=y_size)? (y_num-1) : (int)floor(ray_pos_y/point_step));
			ind_m=(ray_pos_z<point_step)? 0 : (((ray_pos_z+point_step)>=z_size)? (z_num-1) : (int)floor(ray_pos_z/point_step));

			/* initialize misc */
			ind_k_step=((int)(ray_dir_x>=0.0f)<<1)-1;
			ind_l_step=((int)(ray_dir_y>=0.0f)<<1)-1;
			ind_m_step=((int)(ray_dir_z>=0.0f)<<1)-1;
			to_centre_x=point_step*(0.5f+(float)ind_k)-ray_pos_x_const;
			to_centre_y=point_step*(0.5f+(float)ind_l)-ray_pos_y_const;
			to_centre_z=point_step*(0.5f+(float)ind_m)-ray_pos_z_const;
			ind_step_flt_x=point_step*((float)ind_k_step);
			ind_step_flt_y=point_step*((float)ind_l_step);
			ind_step_flt_z=point_step*((float)ind_m_step);
			t_max_x=(to_centre_x+0.5f*ind_step_flt_x)/ray_dir_x;
			t_max_y=(to_centre_y+0.5f*ind_step_flt_y)/ray_dir_y;
			t_max_z=(to_centre_z+0.5f*ind_step_flt_z)/ray_dir_z;
			t_delta_x=ind_step_flt_x/ray_dir_x;
			t_delta_y=ind_step_flt_y/ray_dir_y;
			t_delta_z=ind_step_flt_z/ray_dir_z;

			xy_num=x_num*y_num*ind_m_step;
			yl_num=x_num*ind_l_step;

			mtr_indx=(ind_m*y_num+ind_l)*x_num+ind_k;

			main_color_g=0.0f;
			main_color_r=0.0f;
			this_point_intense_g=1.0f;
			this_point_intense_r=1.0f;

			/* traverse */
			for ( ; ; )
			{
				/* shading */
				ttime=to_centre_x*ray_dir_x+to_centre_y*ray_dir_y+to_centre_z*ray_dir_z;
				t=to_centre_x*to_centre_x+to_centre_y*to_centre_y+(to_centre_z-ttime)*(to_centre_z+ttime);
				if (t<point_radius_sqr)
				{
					t=(point_radius_sqr-t)/point_radius_sqr; // fading of intensity (~1/r^2)
					color_g=GET_G(matrix[mtr_indx]);
					color_r=GET_R(matrix[mtr_indx]);
					if ((color_g>=min_green) && (color_g<=max_green) && (color_r>=min_red) && (color_r<=max_red))
					{
						self_intense=((float)color_g)*t;
						main_color_g+=(this_point_intense_g*self_intense);
						this_point_intense_g*=((255.0f-self_intense)*0.0039216f);
						self_intense=((float)color_r)*t;
						main_color_r+=(this_point_intense_r*self_intense);
						this_point_intense_r*=((255.0f-self_intense)*0.0039216f);
						if ((main_color_g>253.0f) && (main_color_r>253.0f)) break;
					}
				}

				/* further movement */
				if (t_max_x<=t_max_y)
				{
					if (t_max_x<=t_max_z)
					{
						ind_k+=ind_k_step;
						if ((ind_k>=x_num) || (ind_k<0)) break;
						t_max_x+=t_delta_x;
						mtr_indx+=ind_k_step;
						to_centre_x+=ind_step_flt_x;
					}
					else
					{
						ind_m+=ind_m_step;
						if ((ind_m>=z_num) || (ind_m<0)) break;
						t_max_z+=t_delta_z;
						mtr_indx+=xy_num;
						to_centre_z+=ind_step_flt_z;
					}
				}
				else
				{
					if (t_max_y<=t_max_z)
					{
						ind_l+=ind_l_step;
						if ((ind_l>=y_num) || (ind_l<0)) break;
						t_max_y+=t_delta_y;
						mtr_indx+=yl_num;
						to_centre_y+=ind_step_flt_y;
					}
					else
					{
						ind_m+=ind_m_step;
						if ((ind_m>=z_num) || (ind_m<0)) break;
						t_max_z+=t_delta_z;
						mtr_indx+=xy_num;
						to_centre_z+=ind_step_flt_z;
					}
				}
			}

			if ((main_color_g==0.0f) && (main_color_r==0.0f))
				pixels[i*img_w+j]=0xff969696;
			else
				pixels[i*img_w+j]=0xff000000 | 
							((main_color_g>=255.0f)? 0x0000ff00 : (((unsigned int)floor(main_color_g+0.5f))<<8)) | 
							((main_color_r>=255.0f)? 0x00ff0000 : (((unsigned int)floor(main_color_r+0.5f))<<16));
		}
	}
	}
}

void RendererOMP::SelectPoints_by_click (const int x, const int y, const unsigned short *const clr_matrix, 
										 Coords* &pos, int &points_num) const {
	/* see original algorithm in 6 functions above */

	pos=NULL;
	points_num=0;

	const float h_sizes_x=((float)x_num)*point_step*0.5f,h_sizes_y=((float)y_num)*point_step*0.5f,h_sizes_z=((float)z_num)*point_step*0.5f;
	const float ray_pos_const_x=(cam_x-bbox_cnt_x)*ox_x+(cam_y-bbox_cnt_y)*ox_y+(cam_z-bbox_cnt_z)*ox_z;
	const float ray_pos_const_y=(cam_x-bbox_cnt_x)*oy_x+(cam_y-bbox_cnt_y)*oy_y+(cam_z-bbox_cnt_z)*oy_z;
	const float ray_pos_const_z=(cam_x-bbox_cnt_x)*oz_x+(cam_y-bbox_cnt_y)*oz_y+(cam_z-bbox_cnt_z)*oz_z;
	const float trans_x=static_cast<const float>(x+x-static_cast<int>(img_w));
	const float trans_y=static_cast<const float>(y+y-static_cast<int>(img_h));
	float trans_z=(1.0f+M_SQRT2)*static_cast<float>(img_w);
	float ray_dir_x=trans_x*ox_x+trans_y*ox_y+trans_z*ox_z;
	float ray_dir_y=trans_x*oy_x+trans_y*oy_y+trans_z*oy_z;
	float ray_dir_z=trans_x*oz_x+trans_y*oz_y+trans_z*oz_z;

	trans_z=1.0f/sqrtf(ray_dir_x*ray_dir_x+ray_dir_y*ray_dir_y+ray_dir_z*ray_dir_z);
	ray_dir_x*=trans_z;
	ray_dir_y*=trans_z;
	ray_dir_z*=trans_z;

	float r_p_div_p_st_x,r_p_div_p_st_y,r_p_div_p_st_z;

	if ((ray_pos_const_x>h_sizes_x) || (ray_pos_const_x<-h_sizes_x) || 
		(ray_pos_const_y>h_sizes_y) || (ray_pos_const_y<-h_sizes_y) || 
		(ray_pos_const_z>h_sizes_z) || (ray_pos_const_z<-h_sizes_z))
	{
		float ttime=FLT_MAX,t,coord;

		if ((ray_dir_z>FLT_EPSILON) && (ray_pos_const_z<=-h_sizes_z))
		{
			t=-(h_sizes_z+ray_pos_const_z)/ray_dir_z;
			if (t<ttime)
			{
				coord=ray_pos_const_x+ray_dir_x*t;
				if ((coord<=h_sizes_x) && (coord>=-h_sizes_x))
				{
					coord=ray_pos_const_y+ray_dir_y*t;
					if ((coord<=h_sizes_y) && (coord>=-h_sizes_y)) ttime=t;
				}
			}
		}
		else
			if ((ray_dir_z<-FLT_EPSILON) && (ray_pos_const_z>=h_sizes_z))
			{
				t=(h_sizes_z-ray_pos_const_z)/ray_dir_z;
				if (t<ttime)
				{
					coord=ray_pos_const_x+ray_dir_x*t;
					if ((coord<=h_sizes_x) && (coord>=-h_sizes_x))
					{
						coord=ray_pos_const_y+ray_dir_y*t;
						if ((coord<=h_sizes_y) && (coord>=-h_sizes_y)) ttime=t;
					}
				}
			}
		if ((ray_dir_y>FLT_EPSILON) && (ray_pos_const_y<=-h_sizes_y))
		{
			t=-(h_sizes_y+ray_pos_const_y)/ray_dir_y;
			if (t<ttime)
			{
				coord=ray_pos_const_x+ray_dir_x*t;
				if ((coord<=h_sizes_x) && (coord>=-h_sizes_x))
				{
					coord=ray_pos_const_z+ray_dir_z*t;
					if ((coord<=h_sizes_z) && (coord>=-h_sizes_z)) ttime=t;
				}
			}
		}
		else
			if ((ray_dir_y<-FLT_EPSILON) && (ray_pos_const_y>=h_sizes_y))
			{
				t=(h_sizes_y-ray_pos_const_y)/ray_dir_y;
				if (t<ttime)
				{
					coord=ray_pos_const_x+ray_dir_x*t;
					if ((coord<=h_sizes_x) && (coord>=-h_sizes_x))
					{
						coord=ray_pos_const_z+ray_dir_z*t;
						if ((coord<=h_sizes_z) && (coord>=-h_sizes_z)) ttime=t;
					}
				}
			}
		if ((ray_dir_x>FLT_EPSILON) && (ray_pos_const_x<=-h_sizes_x))
		{
			t=-(h_sizes_x+ray_pos_const_x)/ray_dir_x;
			if (t<ttime)
			{
				coord=ray_pos_const_y+ray_dir_y*t;
				if ((coord<=h_sizes_y) && (coord>=-h_sizes_y))
				{
					coord=ray_pos_const_z+ray_dir_z*t;
					if ((coord<=h_sizes_z) && (coord>=-h_sizes_z)) ttime=t;
				}
			}
		}
		else
			if ((ray_dir_x<-FLT_EPSILON) && (ray_pos_const_x>=h_sizes_x))
			{
				t=(h_sizes_x-ray_pos_const_x)/ray_dir_x;
				if (t<ttime)
				{
					coord=ray_pos_const_y+ray_dir_y*t;
					if ((coord<=h_sizes_y) && (coord>=-h_sizes_y))
					{
						coord=ray_pos_const_z+ray_dir_z*t;
						if ((coord<=h_sizes_z) && (coord>=-h_sizes_z)) ttime=t;
					}
				}
			}
		if (ttime>1.0e+29f) return;
		r_p_div_p_st_x=(ray_dir_x*ttime+ray_pos_const_x+h_sizes_x)/point_step;
		r_p_div_p_st_y=(ray_dir_y*ttime+ray_pos_const_y+h_sizes_y)/point_step;
		r_p_div_p_st_z=(ray_dir_z*ttime+ray_pos_const_z+h_sizes_z)/point_step;
	}
	else
	{
		r_p_div_p_st_x=(ray_pos_const_x+h_sizes_x)/point_step;
		r_p_div_p_st_y=(ray_pos_const_y+h_sizes_y)/point_step;
		r_p_div_p_st_z=(ray_pos_const_z+h_sizes_z)/point_step;
	}

	int points_mem_num=10;

	pos=static_cast<Coords*>(malloc(static_cast<size_t>(points_mem_num)*sizeof(Coords)));

	const float p_st_div_r_d_x=point_step/ray_dir_x,p_st_div_r_d_y=point_step/ray_dir_y;
	const float p_st_div_r_d_z=point_step/ray_dir_z;
	int ind_x=(r_p_div_p_st_x<1.0f)? 0 : ((r_p_div_p_st_x>=(float)(x_num-1))? (x_num-1) : (int)floor(r_p_div_p_st_x));
	int ind_y=(r_p_div_p_st_y<1.0f)? 0 : ((r_p_div_p_st_y>=(float)(y_num-1))? (y_num-1) : (int)floor(r_p_div_p_st_y));
	int ind_z=(r_p_div_p_st_z<1.0f)? 0 : ((r_p_div_p_st_z>=(float)(z_num-1))? (z_num-1) : (int)floor(r_p_div_p_st_z));
	const int ind_step_x=(ray_dir_x<0.0f)? -1 : 1;
	const int ind_step_y=(ray_dir_y<0.0f)? -1 : 1;
	const int ind_step_z=(ray_dir_z<0.0f)? -1 : 1;
	float t_max_x=p_st_div_r_d_x*((float)(ind_x+((ind_step_x+1)>>1u))-r_p_div_p_st_x);
	float t_max_y=p_st_div_r_d_y*((float)(ind_y+((ind_step_y+1)>>1u))-r_p_div_p_st_y);
	float t_max_z=p_st_div_r_d_z*((float)(ind_z+((ind_step_z+1)>>1u))-r_p_div_p_st_z);
	const float t_delta_x=p_st_div_r_d_x*(float)ind_step_x;
	const float t_delta_y=p_st_div_r_d_y*(float)ind_step_y;
	const float t_delta_z=p_st_div_r_d_z*(float)ind_step_z;
	const int xy_num=x_num*y_num*ind_step_z;
	const int yl_num=x_num*ind_step_y;
	int mtr_indx=(ind_z*y_num+ind_y)*x_num+ind_x;

	if ((RendBox!=&RendererOMP::RenderBox_cubes_vol) && 
		(RendBox!=&RendererOMP::RenderBox_spheres_vol) && 
		(RendBox!=&RendererOMP::RenderBox_lights_vol))
	{
		// 'volume' mode is off

		const int ind_init_x=ind_x,ind_init_y=ind_y,ind_init_z=ind_z;

		for ( ; ; )
		{
			if ((clr_matrix==NULL) || (clr_matrix[mtr_indx]!=0u))
			{
				if (points_num==points_mem_num)
				{
					points_mem_num<<=1u; // double memory consumption
					pos=static_cast<Coords*>(realloc(pos,static_cast<size_t>(points_mem_num)*sizeof(Coords)));
				}
				pos[points_num].x=ind_x;
				pos[points_num].y=ind_y;
				pos[points_num].z=ind_z;
				++points_num;
			}
			if (t_max_x<=t_max_y)
			{
				if (t_max_x<=t_max_z)
				{
					ind_x+=ind_step_x;
					if ((ind_x>=x_num) || (ind_x<0) || (ind_init_x>ind_x+depth_constraint) || 
						(ind_init_x+depth_constraint<ind_x)) break;
					t_max_x+=t_delta_x;
					mtr_indx+=ind_step_x;
				}
				else
				{
					ind_z+=ind_step_z;
					if ((ind_z>=z_num) || (ind_z<0) || (ind_init_z>ind_z+depth_constraint) || 
						(ind_init_z+depth_constraint<ind_z)) break;
					t_max_z+=t_delta_z;
					mtr_indx+=xy_num;
				}
			}
			else
			{
				if (t_max_y<=t_max_z)
				{
					ind_y+=ind_step_y;
					if ((ind_y>=y_num) || (ind_y<0) || (ind_init_y>ind_y+depth_constraint) || 
						(ind_init_y+depth_constraint<ind_y)) break;
					t_max_y+=t_delta_y;
					mtr_indx+=yl_num;
				}
				else
				{
					ind_z+=ind_step_z;
					if ((ind_z>=z_num) || (ind_z<0) || (ind_init_z>ind_z+depth_constraint) || 
						(ind_init_z+depth_constraint<ind_z)) break;
					t_max_z+=t_delta_z;
					mtr_indx+=xy_num;
				}
			}
		}
	}
	else
	{
		// 'volume' mode is on

		for ( ; ; )
		{
			if ((clr_matrix==NULL) || (clr_matrix[mtr_indx]!=0u))
			{
				if (points_num==points_mem_num)
				{
					points_mem_num<<=1u; // double memory consumption
					pos=static_cast<Coords*>(realloc(pos,static_cast<size_t>(points_mem_num*sizeof(Coords))));
				}
				pos[points_num].x=ind_x;
				pos[points_num].y=ind_y;
				pos[points_num].z=ind_z;
				++points_num;
			}
			if (t_max_x<=t_max_y)
			{
				if (t_max_x<=t_max_z)
				{
					ind_x+=ind_step_x;
					if ((ind_x>=x_num) || (ind_x<0)) break;
					t_max_x+=t_delta_x;
					mtr_indx+=ind_step_x;
				}
				else
				{
					ind_z+=ind_step_z;
					if ((ind_z>=z_num) || (ind_z<0)) break;
					t_max_z+=t_delta_z;
					mtr_indx+=xy_num;
				}
			}
			else
			{
				if (t_max_y<=t_max_z)
				{
					ind_y+=ind_step_y;
					if ((ind_y>=y_num) || (ind_y<0)) break;
					t_max_y+=t_delta_y;
					mtr_indx+=yl_num;
				}
				else
				{
					ind_z+=ind_step_z;
					if ((ind_z>=z_num) || (ind_z<0)) break;
					t_max_z+=t_delta_z;
					mtr_indx+=xy_num;
				}
			}
		}
	}
}

bool RendererOMP::SelectPoints_last_phase (const int x, const int y, Coords &pos, 
										   const unsigned short *const clr_matrix) {
	/* see original algorithm in 6 functions above */

	const float h_sizes_x=((float)x_num)*point_step*0.5f,h_sizes_y=((float)y_num)*point_step*0.5f;
	const float h_sizes_z=((float)z_num)*point_step*0.5f;
	const float ray_pos_const_x=(cam_x-bbox_cnt_x)*ox_x+(cam_y-bbox_cnt_y)*ox_y+(cam_z-bbox_cnt_z)*ox_z;
	const float ray_pos_const_y=(cam_x-bbox_cnt_x)*oy_x+(cam_y-bbox_cnt_y)*oy_y+(cam_z-bbox_cnt_z)*oy_z;
	const float ray_pos_const_z=(cam_x-bbox_cnt_x)*oz_x+(cam_y-bbox_cnt_y)*oz_y+(cam_z-bbox_cnt_z)*oz_z;
	const float trans_x=static_cast<const float>(x+x-static_cast<int>(img_w));
	const float trans_y=static_cast<const float>(y+y-static_cast<int>(img_h));
	float trans_z=(1.0f+M_SQRT2)*static_cast<float>(img_w);
	float ray_dir_x=trans_x*ox_x+trans_y*ox_y+trans_z*ox_z;
	float ray_dir_y=trans_x*oy_x+trans_y*oy_y+trans_z*oy_z;
	float ray_dir_z=trans_x*oz_x+trans_y*oz_y+trans_z*oz_z;

	trans_z=1.0f/sqrtf(ray_dir_x*ray_dir_x+ray_dir_y*ray_dir_y+ray_dir_z*ray_dir_z);
	ray_dir_x*=trans_z;
	ray_dir_y*=trans_z;
	ray_dir_z*=trans_z;

	float r_p_div_p_st_x,r_p_div_p_st_y,r_p_div_p_st_z;

	if ((ray_pos_const_x>h_sizes_x) || (ray_pos_const_x<-h_sizes_x) || 
		(ray_pos_const_y>h_sizes_y) || (ray_pos_const_y<-h_sizes_y) || 
		(ray_pos_const_z>h_sizes_z) || (ray_pos_const_z<-h_sizes_z))
	{
		float ttime=FLT_MAX,t,coord;

		if ((ray_dir_z>FLT_EPSILON) && (ray_pos_const_z<=-h_sizes_z))
		{
			t=-(h_sizes_z+ray_pos_const_z)/ray_dir_z;
			if (t<ttime)
			{
				coord=ray_pos_const_x+ray_dir_x*t;
				if ((coord<=h_sizes_x) && (coord>=-h_sizes_x))
				{
					coord=ray_pos_const_y+ray_dir_y*t;
					if ((coord<=h_sizes_y) && (coord>=-h_sizes_y)) ttime=t;
				}
			}
		}
		else
			if ((ray_dir_z<-FLT_EPSILON) && (ray_pos_const_z>=h_sizes_z))
			{
				t=(h_sizes_z-ray_pos_const_z)/ray_dir_z;
				if (t<ttime)
				{
					coord=ray_pos_const_x+ray_dir_x*t;
					if ((coord<=h_sizes_x) && (coord>=-h_sizes_x))
					{
						coord=ray_pos_const_y+ray_dir_y*t;
						if ((coord<=h_sizes_y) && (coord>=-h_sizes_y)) ttime=t;
					}
				}
			}
		if ((ray_dir_y>FLT_EPSILON) && (ray_pos_const_y<=-h_sizes_y))
		{
			t=-(h_sizes_y+ray_pos_const_y)/ray_dir_y;
			if (t<ttime)
			{
				coord=ray_pos_const_x+ray_dir_x*t;
				if ((coord<=h_sizes_x) && (coord>=-h_sizes_x))
				{
					coord=ray_pos_const_z+ray_dir_z*t;
					if ((coord<=h_sizes_z) && (coord>=-h_sizes_z)) ttime=t;
				}
			}
		}
		else
			if ((ray_dir_y<-FLT_EPSILON) && (ray_pos_const_y>=h_sizes_y))
			{
				t=(h_sizes_y-ray_pos_const_y)/ray_dir_y;
				if (t<ttime)
				{
					coord=ray_pos_const_x+ray_dir_x*t;
					if ((coord<=h_sizes_x) && (coord>=-h_sizes_x))
					{
						coord=ray_pos_const_z+ray_dir_z*t;
						if ((coord<=h_sizes_z) && (coord>=-h_sizes_z)) ttime=t;
					}
				}
			}
		if ((ray_dir_x>FLT_EPSILON) && (ray_pos_const_x<=-h_sizes_x))
		{
			t=-(h_sizes_x+ray_pos_const_x)/ray_dir_x;
			if (t<ttime)
			{
				coord=ray_pos_const_y+ray_dir_y*t;
				if ((coord<=h_sizes_y) && (coord>=-h_sizes_y))
				{
					coord=ray_pos_const_z+ray_dir_z*t;
					if ((coord<=h_sizes_z) && (coord>=-h_sizes_z)) ttime=t;
				}
			}
		}
		else
			if ((ray_dir_x<-FLT_EPSILON) && (ray_pos_const_x>=h_sizes_x))
			{
				t=(h_sizes_x-ray_pos_const_x)/ray_dir_x;
				if (t<ttime)
				{
					coord=ray_pos_const_y+ray_dir_y*t;
					if ((coord<=h_sizes_y) && (coord>=-h_sizes_y))
					{
						coord=ray_pos_const_z+ray_dir_z*t;
						if ((coord<=h_sizes_z) && (coord>=-h_sizes_z)) ttime=t;
					}
				}
			}
		if (ttime>1.0e+29f) return false;
		r_p_div_p_st_x=(ray_dir_x*ttime+ray_pos_const_x+h_sizes_x)/point_step;
		r_p_div_p_st_y=(ray_dir_y*ttime+ray_pos_const_y+h_sizes_y)/point_step;
		r_p_div_p_st_z=(ray_dir_z*ttime+ray_pos_const_z+h_sizes_z)/point_step;
	}
	else
	{
		r_p_div_p_st_x=(ray_pos_const_x+h_sizes_x)/point_step;
		r_p_div_p_st_y=(ray_pos_const_y+h_sizes_y)/point_step;
		r_p_div_p_st_z=(ray_pos_const_z+h_sizes_z)/point_step;
	}

	const float p_st_div_r_d_x=point_step/ray_dir_x,p_st_div_r_d_y=point_step/ray_dir_y;
	const float p_st_div_r_d_z=point_step/ray_dir_z;
	int ind_x=(r_p_div_p_st_x<1.0f)? 0 : ((r_p_div_p_st_x>=(float)(x_num-1))? (x_num-1) : (int)floor(r_p_div_p_st_x));
	int ind_y=(r_p_div_p_st_y<1.0f)? 0 : ((r_p_div_p_st_y>=(float)(y_num-1))? (y_num-1) : (int)floor(r_p_div_p_st_y));
	int ind_z=(r_p_div_p_st_z<1.0f)? 0 : ((r_p_div_p_st_z>=(float)(z_num-1))? (z_num-1) : (int)floor(r_p_div_p_st_z));
	const int ind_step_x=(ray_dir_x<0.0f)? -1 : 1;
	const int ind_step_y=(ray_dir_y<0.0f)? -1 : 1;
	const int ind_step_z=(ray_dir_z<0.0f)? -1 : 1;
	float t_max_x=p_st_div_r_d_x*((float)(ind_x+((ind_step_x+1)>>1u))-r_p_div_p_st_x);
	float t_max_y=p_st_div_r_d_y*((float)(ind_y+((ind_step_y+1)>>1u))-r_p_div_p_st_y);
	float t_max_z=p_st_div_r_d_z*((float)(ind_z+((ind_step_z+1)>>1u))-r_p_div_p_st_z);
	const float t_delta_x=p_st_div_r_d_x*(float)ind_step_x;
	const float t_delta_y=p_st_div_r_d_y*(float)ind_step_y;
	const float t_delta_z=p_st_div_r_d_z*(float)ind_step_z;
	const int xy_num=x_num*y_num*ind_step_z;
	const int yl_num=x_num*ind_step_y;
	int mtr_indx=(ind_z*y_num+ind_y)*x_num+ind_x;
	float to_centre_x,to_centre_y,to_centre_z;
	const float point_step_sq=point_step*point_step,point_radius_sqr=point_radius*point_radius;
	bool was_hit;

	for ( ; ; )
	{
		if ((RendBox==&RendererOMP::RenderBox_cubes) || (RendBox==&RendererOMP::RenderBox_cubes_vol))
			was_hit=true;
		else
		{
			to_centre_x=((float)ind_x+0.5f-r_p_div_p_st_x);
			to_centre_y=((float)ind_y+0.5f-r_p_div_p_st_y);
			to_centre_z=((float)ind_z+0.5f-r_p_div_p_st_z);
			trans_z=to_centre_x*ray_dir_x+to_centre_y*ray_dir_y+to_centre_z*ray_dir_z;
			was_hit=(point_radius_sqr>=point_step_sq*((to_centre_x-trans_z)*(to_centre_x+trans_z)+\
			to_centre_y*to_centre_y+to_centre_z*to_centre_z));
		}
		if (was_hit && (clr_matrix[mtr_indx]!=0u))
		{
			pos.x=ind_x;
			pos.y=ind_y;
			pos.z=ind_z;

			pt_selection_mode=true;
			sel_ind_x=ind_x;
			sel_ind_y=ind_y;
			sel_ind_z=ind_z;
			BuildSelectionCube(ray_pos_const_x+h_sizes_x,ray_pos_const_y+h_sizes_y,ray_pos_const_z+h_sizes_z);
			return true;
		}
		if (t_max_x<=t_max_y)
		{
			if (t_max_x<=t_max_z)
			{
				ind_x+=ind_step_x;
				if ((ind_x>=x_num) || (ind_x<0)) break;
				t_max_x+=t_delta_x;
				mtr_indx+=ind_step_x;
			}
			else
			{
				ind_z+=ind_step_z;
				if ((ind_z>=z_num) || (ind_z<0)) break;
				t_max_z+=t_delta_z;
				mtr_indx+=xy_num;
			}
		}
		else
		{
			if (t_max_y<=t_max_z)
			{
				ind_y+=ind_step_y;
				if ((ind_y>=y_num) || (ind_y<0)) break;
				t_max_y+=t_delta_y;
				mtr_indx+=yl_num;
			}
			else
			{
				ind_z+=ind_step_z;
				if ((ind_z>=z_num) || (ind_z<0)) break;
				t_max_z+=t_delta_z;
				mtr_indx+=xy_num;
			}
		}
	}
	return false;
}

void RendererOMP::BuildSelectionCube (const float ray_pos_x, const float ray_pos_y, const float ray_pos_z) const {
	/* don't forget that all calculations are carried in bounding box coordinate system! */

	memset(fv_sel_cube_vis,0,8u*sizeof(bool));

	float to_centre_x=point_step*static_cast<float>(sel_ind_x+2)-ray_pos_x;
	float to_centre_y=point_step*static_cast<float>(sel_ind_y+2)-ray_pos_y;
	float to_centre_z=point_step*static_cast<float>(sel_ind_z+2)-ray_pos_z;

	/* determine visibility of 8 corner points of selection cube (5 big steps below) */

	if ((to_centre_x<=point_step+point_step) && (point_step<=to_centre_x) && 
		(to_centre_y<=point_step+point_step) && (point_step<=to_centre_y) && 
		(to_centre_z<=point_step+point_step) && (point_step<=to_centre_z))
	// 'ray_pos' is inside the cube - assume the whole cube invisible
	return;

	const float trans_z=(1.0f+M_SQRT2)*static_cast<const float>(img_w);
	const float img_w_2=0.5f*static_cast<const float>(img_w),img_h_2=0.5f*static_cast<const float>(img_h);
	float a,b,c,d,e,f=-1.5f*point_step,mlt;
	bool hit=false;

	/* STEP 1: transform centre of selected "point" into screen coordinates 
	           ('d' will be (j-img_w/2), 'e' will be (i-img_h/2)) */
	a=(to_centre_x+f)*oz_y-(to_centre_z+f)*ox_y;
	b=(to_centre_x+f)*oy_x-(to_centre_y+f)*ox_x;
	c=(to_centre_x+f)*oz_x-(to_centre_z+f)*ox_x;
	d=(to_centre_x+f)*oy_y-(to_centre_y+f)*ox_y;
	mlt=a*b-c*d;
	if ((mlt<FLT_EPSILON) && (mlt>-FLT_EPSILON))
	{
		a=(to_centre_y+f)*oz_y-(to_centre_z+f)*oy_y;
		c=(to_centre_y+f)*oz_x-(to_centre_z+f)*oy_x;
		mlt=a*b-c*d;
		if ((mlt<FLT_EPSILON) && (mlt>-FLT_EPSILON))
		{
			mlt=trans_z*ox_z/oz_z;
			d=floor(img_w_2+(oy_y-ox_y)*mlt+0.5f)-img_w_2;
			e=floor(img_h_2+(ox_x-oy_x)*mlt+0.5f)-img_h_2;
		}
		else
		{
			e=(to_centre_y+f)*oz_z-(to_centre_z+f)*oy_z;
			f=(to_centre_x+f)*oy_z-(to_centre_y+f)*ox_z;
			mlt=trans_z/(mlt+mlt);
			d=floor((e*d-a*f)*mlt+img_w_2+0.5f)-img_w_2;
			e=floor((c*f-e*b)*mlt+img_h_2+0.5f)-img_h_2;
		}
	}
	else
	{
		e=(to_centre_x+f)*oz_z-(to_centre_z+f)*ox_z;
		f=(to_centre_x+f)*oy_z-(to_centre_y+f)*ox_z;
		mlt=trans_z/(mlt+mlt);
		d=floor((e*d-a*f)*mlt+img_w_2+0.5f)-img_w_2;
		e=floor((c*f-e*b)*mlt+img_h_2+0.5f)-img_h_2;
	}
	if ((d<=-img_w_2) || (d>=img_w_2) || (e<=-img_h_2) || (e>=img_h_2))
		// centre of selected "point" is invisible - assume the whole cube invisible
		return;
	/* compute a ray from 'ray_pos' to found coordinates */
	f=0.5f*trans_z;
	a=d*ox_x+e*ox_y+f*ox_z;
	b=d*oy_x+e*oy_y+f*oy_z;
	c=d*oz_x+e*oz_y+f*oz_z;
	// no need in normalization of 'a', 'b' and 'c' (aka 'ray_dir')

	/* STEP 2: find intersection of the ray and faces of selected "point" */
	if ((c>FLT_EPSILON) && (point_step+point_step<=to_centre_z))
	{
		e=(to_centre_z-point_step-point_step)/c;
		mlt=a*e+point_step;
		if ((mlt+point_step>=to_centre_x) && (mlt<=to_centre_x))
		{
			mlt=b*e+point_step;
			if ((mlt+point_step>=to_centre_y) && (mlt<=to_centre_y)) hit=true;
		}
	}
	else
		if ((c<-FLT_EPSILON) && (point_step>=to_centre_z))
		{
			e=(to_centre_z-point_step)/c;
			mlt=a*e+point_step;
			if ((mlt+point_step>=to_centre_x) && (mlt<=to_centre_x))
			{
				mlt=b*e+point_step;
				if ((mlt+point_step>=to_centre_y) && (mlt<=to_centre_y)) hit=true;
			}
		}
	if (!hit)
	{
		if ((b>FLT_EPSILON) && (point_step+point_step<=to_centre_y))
		{
			e=(to_centre_y-point_step-point_step)/b;
			mlt=a*e+point_step;
			if ((mlt+point_step>=to_centre_x) && (mlt<=to_centre_x))
			{
				mlt=c*e+point_step;
				if ((mlt+point_step>=to_centre_z) && (mlt<=to_centre_z)) hit=true;
			}
		}
		else
			if ((b<-FLT_EPSILON) && (point_step>=to_centre_y))
			{
				e=(to_centre_y-point_step)/b;
				mlt=a*e+point_step;
				if ((mlt+point_step>=to_centre_x) && (mlt<=to_centre_x))
				{
					mlt=c*e+point_step;
					if ((mlt+point_step>=to_centre_z) && (mlt<=to_centre_z)) hit=true;
				}
			}
		if (!hit)
		{
			if ((a>FLT_EPSILON) && (point_step+point_step<=to_centre_x))
			{
				e=(to_centre_x-point_step-point_step)/a;
				mlt=b*e+point_step;
				if ((mlt+point_step>=to_centre_y) && (mlt<=to_centre_y))
				{
					mlt=c*e+point_step;
					if ((mlt+point_step>=to_centre_z) && (mlt<=to_centre_z)) hit=true;
				}
			}
			else
				if ((a<-FLT_EPSILON) && (point_step>=to_centre_x))
				{
					e=(to_centre_x-point_step)/a;
					mlt=b*e+point_step;
					if ((mlt+point_step>=to_centre_y) && (mlt<=to_centre_y))
					{
						mlt=c*e+point_step;
						if ((mlt+point_step>=to_centre_z) && (mlt<=to_centre_z)) hit=true;
					}
				}
		}
	}
	if (!hit) return; // no intersection - the selected "point" itself is invisible

	char i,j,k;
	int ind=0;

	/* STEP 3: find the nearest point of the cube to 'ray_pos' */
	a=FLT_MAX;
	to_centre_z-=(point_step+point_step);
	for (k=0; k<8; k+=4)
	{
		d=to_centre_z*to_centre_z;
		to_centre_y-=(point_step+point_step);
		for (j=0; j<2; ++j)
		{
			c=to_centre_y*to_centre_y+d;
			to_centre_x-=(point_step+point_step);
			for (i=0; i<2; ++i)
			{
				b=to_centre_x*to_centre_x+c;
				if (b<a)
				{
					a=b;
					ind=k+j+j+i;
				}
				to_centre_x+=point_step;
			}
			to_centre_y+=point_step;
		}
		to_centre_z+=point_step;
	}

	/* one point belongs to 3 faces; 
	   maximum of 3 faces are visible simultaneously in a cube from 'ray_pos'; 
	   the nearest point found above must belong to all these faces (I don't know how to prove this); 
	   but one of these faces can make another invisible. 
	   STEP 4: Let's check it */
	char faces[3]; // possibly visible faces
	bool vis[3]={true,true,true}; // indicator of visibility

	/* numbering of faces: 
	   0 - all 4 points have "near" z-coordinate,
	   1 - all 4 points have "bottom" y-coordinate,
	   2 - all 4 points have "left" x-coordinate,
	   3 - all 4 points have "far" z-coordinate,
	   4 - all 4 points have "top" y-coordinate,
	   5 - all 4 points have "right" x-coordinate */
	switch (ind)
	{
		case 0: faces[0]=0; faces[1]=1; faces[2]=2; break;
		case 1: faces[0]=0; faces[1]=1; faces[2]=5; break;
		case 2: faces[0]=0; faces[1]=2; faces[2]=4; break;
		case 3: faces[0]=0; faces[1]=4; faces[2]=5; break;
		case 4: faces[0]=1; faces[1]=2; faces[2]=3; break;
		case 5: faces[0]=1; faces[1]=3; faces[2]=5; break;
		case 6: faces[0]=2; faces[1]=3; faces[2]=4; break;
		case 7: faces[0]=3; faces[1]=4; faces[2]=5; break;
	}
	f=-1.5f*point_step;
	for (ind=0; ind<3; ++ind)
	{
		switch (faces[ind])
		{
			case 0:
				a=to_centre_x+f;
				b=to_centre_y+f;
				c=to_centre_z-point_step-point_step;
				break;
			case 1:
				a=to_centre_x+f;
				b=to_centre_y-point_step-point_step;
				c=to_centre_z+f;
				break;
			case 2:
				a=to_centre_x-point_step-point_step;
				b=to_centre_y+f;
				c=to_centre_z+f;
				break;
			case 3:
				a=to_centre_x+f;
				b=to_centre_y+f;
				c=to_centre_z-point_step;
				break;
			case 4:
				a=to_centre_x+f;
				b=to_centre_y-point_step;
				c=to_centre_z+f;
				break;
			case 5:
				a=to_centre_x-point_step;
				b=to_centre_y+f;
				c=to_centre_z+f;
				break;
		}
		for (j=1; j<3; ++j)
		{
			i=faces[(ind+static_cast<int>(j))%3];
			mlt=-point_step;
			if (i<3) mlt-=point_step;
			switch (i)
			{
				case 0:
				case 3:
					if ((c<-FLT_EPSILON) || (c>FLT_EPSILON))
					{
						mlt=(to_centre_z+mlt)/c;
						if (mlt>0.0f)
						{
							d=mlt*a+point_step;
							if ((d+point_step>to_centre_x) && (d<to_centre_x))
							{
								d=mlt*b+point_step;
								if ((d+point_step>to_centre_y) && (d<to_centre_y)) vis[ind]=false;
							}
						}
					}
					break;
				case 1:
				case 4:
					if ((b<-FLT_EPSILON) || (b>FLT_EPSILON))
					{
						mlt=(to_centre_y+mlt)/b;
						if (mlt>0.0f)
						{
							d=mlt*a+point_step;
							if ((d+point_step>to_centre_x) && (d<to_centre_x))
							{
								d=mlt*c+point_step;
								if ((d+point_step>to_centre_z) && (d<to_centre_z)) vis[ind]=false;
							}
						}
					}
					break;
				case 2:
				case 5:
					if ((a<-FLT_EPSILON) || (a>FLT_EPSILON))
					{
						mlt=(to_centre_x+mlt)/a;
						if (mlt>0.0f)
						{
							d=mlt*b+point_step;
							if ((d+point_step>to_centre_y) && (d<to_centre_y))
							{
								d=mlt*c+point_step;
								if ((d+point_step>to_centre_z) && (d<to_centre_z)) vis[ind]=false;
							}
						}
					}
					break;
			}
			if (!vis[ind]) break; // no need to check another face
		}
	}

	/* STEP 5: visible faces were found; so all other faces are invisible. 
	           If a face is visible than all its 4 points are visible */
	for (ind=0; ind<3; ++ind)
	{
		if (!vis[ind]) continue; // points of a face 'faces[ind]' are all invisible
		switch (faces[ind])
		{
			case 0: fv_sel_cube_vis[0]=fv_sel_cube_vis[1]=fv_sel_cube_vis[2]=fv_sel_cube_vis[3]=true; break;
			case 1: fv_sel_cube_vis[0]=fv_sel_cube_vis[1]=fv_sel_cube_vis[4]=fv_sel_cube_vis[5]=true; break;
			case 2: fv_sel_cube_vis[0]=fv_sel_cube_vis[2]=fv_sel_cube_vis[4]=fv_sel_cube_vis[6]=true; break;
			case 3: fv_sel_cube_vis[4]=fv_sel_cube_vis[5]=fv_sel_cube_vis[6]=fv_sel_cube_vis[7]=true; break;
			case 4: fv_sel_cube_vis[2]=fv_sel_cube_vis[3]=fv_sel_cube_vis[6]=fv_sel_cube_vis[7]=true; break;
			case 5: fv_sel_cube_vis[1]=fv_sel_cube_vis[3]=fv_sel_cube_vis[5]=fv_sel_cube_vis[7]=true; break;
		}
	}

	/* transform visible points' coordinates to screen coordinate system */
	ind=0;
	to_centre_z-=(point_step+point_step);
	for (k=0; k<2; ++k)
	{
		to_centre_y-=(point_step+point_step);
		for (j=0; j<2; ++j)
		{
			to_centre_x-=(point_step+point_step);
			for (i=0; i<2; ++i)
			{
				if (fv_sel_cube_vis[ind])
				{
					a=to_centre_x*oz_y-to_centre_z*ox_y;
					b=to_centre_x*oy_x-to_centre_y*ox_x;
					c=to_centre_x*oz_x-to_centre_z*ox_x;
					d=to_centre_x*oy_y-to_centre_y*ox_y;
					mlt=a*b-c*d;
					if ((mlt<FLT_EPSILON) && (mlt>-FLT_EPSILON))
					{
						a=to_centre_y*oz_y-to_centre_z*oy_y;
						c=to_centre_y*oz_x-to_centre_z*oy_x;
						mlt=a*b-c*d;
						if ((mlt<FLT_EPSILON) && (mlt>-FLT_EPSILON))
						{
							mlt=trans_z*ox_z/oz_z;
							fv_sel_cube_x[ind]=static_cast<int>(floor(img_w_2+(oy_y-ox_y)*mlt+0.5f));
							fv_sel_cube_y[ind]=static_cast<int>(floor(img_h_2+(ox_x-oy_x)*mlt+0.5f));
							to_centre_x+=point_step;
							++ind;
							continue;
						}
						e=to_centre_y*oz_z-to_centre_z*oy_z;
					}
					else e=to_centre_x*oz_z-to_centre_z*ox_z;
					f=to_centre_x*oy_z-to_centre_y*ox_z;
					mlt=trans_z/(mlt+mlt);
					fv_sel_cube_x[ind]=static_cast<int>(floor((e*d-a*f)*mlt+img_w_2+0.5f));
					fv_sel_cube_y[ind]=static_cast<int>(floor((c*f-e*b)*mlt+img_h_2+0.5f));
				}
				to_centre_x+=point_step;
				++ind;
			}
			to_centre_y+=point_step;
		}
		to_centre_z+=point_step;
	}
}

