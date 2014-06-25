#include "renderer_OpenCL.h"
#include <cstdio>
#include <cstring>
#include <cfloat>

#define KERNEL_NAME_CUBES       "RenderBox_cubes"       // name of kernel (cubes) function
#define KERNEL_NAME_SPHERES     "RenderBox_spheres"     // name of kernel (spheres) function
#define KERNEL_NAME_LIGHTS      "RenderBox_lights"      // name of kernel ("lights") function
#define KERNEL_NAME_CUBES_VOL   "RenderBox_cubes_vol"   // name of kernel (volume; cubes) function
#define KERNEL_NAME_SPHERES_VOL "RenderBox_spheres_vol" // name of kernel (volume; spheres) function
#define KERNEL_NAME_LIGHTS_VOL  "RenderBox_lights_vol"  // name of kernel (volume; "lights") function

//#define MAX_KERNEL_NAME_LEN     22 // the bigest length of the names above plus 1

#define SOURCE_FILE "core/renderer_OpenCL.ocl" // file with code for kernels above

#define ARG_PASS_ERR(num,ret) { this->err_str=QObject::tr("cannot pass the argument %1 to the kernel").arg(num); return ret; }
			    
RendererOCL::RendererOCL (int *const fv_sl_cube_x, int *const fv_sl_cube_y, bool *const fv_sl_cube_vis): 
  fv_sel_cube_x(fv_sl_cube_x), fv_sel_cube_y(fv_sl_cube_y), fv_sel_cube_vis(fv_sl_cube_vis) {
	contxt=NULL;
	program=NULL;
	kernel=NULL;
	comm_queue=NULL;
	gpu_pixels=NULL;
	matrix=NULL;

	cl_platform_id *pl_id;
	cl_device_id dev_id;
	cl_uint n_pl,pl_need;

	clGetPlatformIDs(0/* num_entries */,NULL/* platforms */,&n_pl/* num_platforms */); // only to get the exact number
	pl_id=static_cast<cl_platform_id*>(malloc(n_pl*sizeof(cl_platform_id)));
	clGetPlatformIDs(n_pl,pl_id,NULL);

	for (pl_need=0; pl_need<n_pl; ++pl_need)
	{
		if (clGetDeviceIDs(pl_id[pl_need]/* platform */,CL_DEVICE_TYPE_GPU/* device_type */,
			1/* num_entries */,&dev_id/* devices */,NULL/* num_devices */)==CL_SUCCESS)
			break;
	}
	if (pl_need==n_pl)
	{
		free(pl_id);
		this->err_str=QObject::tr("no suitable device was found");
		return;
	}

	const cl_context_properties cps[]={CL_CONTEXT_PLATFORM,(cl_context_properties)pl_id[pl_need],0};
	contxt=clCreateContext(cps,1,&dev_id,NULL,NULL,NULL);
	free(pl_id);
	if (contxt==NULL)
	{
		this->err_str=QObject::tr("cannot create the context");
		return;
	}

	FILE *src=fopen(SOURCE_FILE,"r"); // source code with kernels!
	if (src==NULL)
	{
		((this->err_str=QObject::tr("source file \""))+=QObject::tr(SOURCE_FILE))+=QObject::tr("\" cannot be read");
		return;
	}
	char **sources=NULL;
	unsigned int src_num=0u,i;
	char buf[256];
	unsigned int str_len;
	if (!feof(src))
	{
		src_num=1u;
		sources=static_cast<char**>(malloc(sizeof(char*)));
		if (fgets(buf,256,src)!=NULL) {}
		sources[0]=static_cast<char*>(malloc((strlen(buf)+1u)*sizeof(char)));
		strcpy(sources[0],buf);
		while (fgets(buf,256,src)!=NULL)
		{
			str_len=strlen(sources[src_num-1u]);
			sources[src_num-1u]=static_cast<char*>(realloc(sources[src_num-1u],
														   (str_len+strlen(buf)+1u)*sizeof(char)));
			strcat(sources[src_num-1u],buf);
			if (buf[strlen(buf)-1u]=='\n')
			{
				sources=static_cast<char**>(realloc(sources,(src_num+1u)*sizeof(char*)));
				sources[src_num]=static_cast<char*>(malloc(sizeof(char)));
				sources[src_num][0]='\0';
				++src_num;
			}
		}
	}
	fclose(src);
	program=clCreateProgramWithSource(contxt,static_cast<cl_uint>(src_num),const_cast<const char**>(sources),NULL,NULL);
	if (program==NULL)
	{
		if (sources!=NULL)
		{
			for (i=0u; i!=src_num; ++i)
				free(sources[i]);
			free(sources);
		}
		this->err_str=QObject::tr("cannot create the program");
		return;
	}
	if (clBuildProgram(program,1/* num_devices */,&dev_id/* device_list */,"-w -Werror"/* options */,NULL,NULL)
		!=CL_SUCCESS)
	{
		size_t build_log_size;
		clGetProgramBuildInfo(program/* program */,dev_id/* device */,CL_PROGRAM_BUILD_LOG/* param_name */,
						0/* param_value_size */,NULL/* param_value */,&build_log_size/* param_value_size_ret */);
		char *build_log=static_cast<char*>(malloc(build_log_size));
		clGetProgramBuildInfo(program/* program */,dev_id/* device */,CL_PROGRAM_BUILD_LOG/* param_name */,
				build_log_size/* param_value_size */,build_log/* param_value */,NULL/* param_value_size_ret */);
		this->err_str=QObject::tr("cannot build the program\n");
		this->err_str+=QObject::tr(build_log);
		free(build_log);
		if (sources!=NULL)
		{
			for (i=0u; i<src_num; ++i)
				free(sources[i]);
			free(sources);
		}
		return;
	}
	if (sources!=NULL)
	{
		for (i=0u; i<src_num; ++i)
			free(sources[i]);
		free(sources);
	}

	kernel=clCreateKernel(program/* program */,KERNEL_NAME_CUBES/* kernel_name */,NULL/* errcode_ret */);
	if (kernel==NULL)
	{
		((this->err_str=QObject::tr("no such kernel named \""))+=KERNEL_NAME_CUBES)+='\"';
		return;
	}

	comm_queue=clCreateCommandQueue(contxt,dev_id,0,NULL);
	if (comm_queue==NULL)
	{
		this->err_str=QObject::tr("cannot create the command queue");
		return;
	}
}

void RendererOCL::Init (const unsigned int image_w, const unsigned int image_h, 
						const int num_x, const int num_y, const int num_z) {  
	img_w=image_w;
	img_h=image_h;

	data.sc=5.0f; // 'point_step'
	data.sd=0.16f*data.sc*data.sc; // == 'point_radius' * 'point_radius'

	pt_repr=CUBES;

	num.x=num_x;
	num.y=num_y;
	num.z=num_z;

	data.s0=1.0f;
	data.s1=0.0f;
	data.s2=0.0f;
	data.s3=0.0f;
	data.s4=-1.0f;
	data.s5=0.0f;
	data.s6=0.0f;
	data.s7=0.0f;
	data.s8=1.0f;

	data.s9=0.0f;
	data.sa=0.0f;
	data.sb=-500.0f-(5200.0f+0.5f*static_cast<float>(num_z)*data.sc);

	depth_constraint=1u<<20u; // some kind of big number

	if (clSetKernelArg(kernel,0/* arg_index */,sizeof(cl_float16)/* arg_size */,&data/* arg_value */)!=CL_SUCCESS)
		ARG_PASS_ERR(0,)
	if (clSetKernelArg(kernel,1,sizeof(cl_int3),&num)!=CL_SUCCESS)
		ARG_PASS_ERR(1,)
	if (clSetKernelArg(kernel,2,sizeof(int),&depth_constraint)!=CL_SUCCESS)
		ARG_PASS_ERR(2,)

	gpu_pixels=clCreateBuffer(contxt,CL_MEM_WRITE_ONLY,image_w*image_h*sizeof(int),NULL,NULL);
	/*cl_image_format img_format;
	img_format.image_channel_order=CL_RGBA;
	img_format.image_channel_data_type=CL_UNSIGNED_INT8;
	gpu_pixels=clCreateImage2D(contxt,CL_MEM_WRITE_ONLY,&img_format,image_w,image_h,0,NULL,NULL);*/
	if (gpu_pixels==NULL)
	{
		this->err_str=QObject::tr("cannot allocate memory for the image on GPU");
		return;
	}
	if (clSetKernelArg(kernel,4,sizeof(gpu_pixels),&gpu_pixels)!=CL_SUCCESS)
		ARG_PASS_ERR(4,)

	clr_minmax.x=0u;
	clr_minmax.y=255u;
	clr_minmax.z=0u;
	clr_minmax.w=255u;

	volume_mode=false;
	pt_selection_mode=false;
}

void RendererOCL::RenderBox (unsigned int *pixels) const {
	clFinish(comm_queue); // to be sure that the queue is empty

	const size_t all_threads_num[]={img_w,img_h}; // IF YOU CHANGE THIS, MODIFY 'renderer_OpenCL.ocl'!!
	static const size_t work_group_threads_num[]={2u,32u}; // 'img_w' must contain the first number and 'img_h' - the second number!!
	clEnqueueNDRangeKernel(comm_queue,kernel,2/*work_dim*/,NULL/*offset*/,all_threads_num,work_group_threads_num,
						   0,NULL,NULL);

	clEnqueueReadBuffer(comm_queue,gpu_pixels,CL_TRUE,0,img_w*img_h*sizeof(int),pixels,0,NULL,NULL);
	//const size_t origin[3]={0u,0u,0u},region[3]={img_w,img_h,1u};
	//clEnqueueReadImage(comm_queue,gpu_pixels,CL_TRUE,origin,region,0/* row_pitch */,0,pixels,0,NULL,NULL);
}

PtRepr RendererOCL::ChangeKernel (const PtRepr pt_rpr) {
	if (pt_repr==pt_rpr) return pt_rpr;

	this->err_str.clear();

	clReleaseKernel(kernel);

	switch (pt_rpr)
	{
		case CUBES:
			kernel=clCreateKernel(program,(volume_mode)? KERNEL_NAME_CUBES_VOL : KERNEL_NAME_CUBES,NULL);
			if (kernel==NULL)
			{
				this->err_str=QObject::tr("no such kernel named \"");
				this->err_str+=(volume_mode? KERNEL_NAME_CUBES_VOL : KERNEL_NAME_CUBES);
				this->err_str+='\"';
				return pt_repr;
			}
			break;
		case SPHERES:
			kernel=clCreateKernel(program,(volume_mode)? KERNEL_NAME_SPHERES_VOL : KERNEL_NAME_SPHERES,NULL);
			if (kernel==NULL)
			{
				this->err_str=QObject::tr("no such kernel named \"");
				this->err_str+=(volume_mode? KERNEL_NAME_SPHERES_VOL : KERNEL_NAME_SPHERES);
				this->err_str+='\"';
				return pt_repr;
			}
			break;
		case LIGHTS:
			kernel=clCreateKernel(program,(volume_mode)? KERNEL_NAME_LIGHTS_VOL : KERNEL_NAME_LIGHTS,NULL);
			if (kernel==NULL)
			{
				this->err_str=QObject::tr("no such kernel named \"");
				this->err_str+=(volume_mode? KERNEL_NAME_LIGHTS_VOL : KERNEL_NAME_LIGHTS);
				this->err_str+='\"';
				return pt_repr;
			}
			break;
		default:
			this->err_str=QObject::tr("unknown representation type of \"points\"");
			return pt_repr;
	}
	if (clSetKernelArg(kernel,0,sizeof(cl_float16),&data)!=CL_SUCCESS)
		ARG_PASS_ERR(0,pt_repr)
	if (clSetKernelArg(kernel,1,sizeof(cl_int3),&num)!=CL_SUCCESS)
		ARG_PASS_ERR(1,pt_repr)
	if (volume_mode)
	{
		if (clSetKernelArg(kernel,2,sizeof(cl_uchar4),&clr_minmax)!=CL_SUCCESS)
			ARG_PASS_ERR(2,pt_repr)
	}
	else
	{
		if (clSetKernelArg(kernel,2,sizeof(int),&depth_constraint)!=CL_SUCCESS)
			ARG_PASS_ERR(2,pt_repr)
	}
	if (clSetKernelArg(kernel,3,sizeof(cl_mem),&matrix)!=CL_SUCCESS)
		ARG_PASS_ERR(3,pt_repr)
	if (clSetKernelArg(kernel,4,sizeof(gpu_pixels),&gpu_pixels)!=CL_SUCCESS)
		ARG_PASS_ERR(4,pt_repr)

	const PtRepr old_pt_repr=pt_repr;
	pt_repr=pt_rpr;
	return old_pt_repr;
}

bool RendererOCL::ToggleVolumeMode (const bool on) {
	if (on==volume_mode) return true; // this state has been already set

	clReleaseKernel(kernel);

	switch (pt_repr)
	{
		case CUBES:
			kernel=clCreateKernel(program,on? KERNEL_NAME_CUBES_VOL : KERNEL_NAME_CUBES,NULL);
			if (kernel==NULL)
			{
				this->err_str=QObject::tr("no such kernel named \"");
				this->err_str+=(on? KERNEL_NAME_CUBES_VOL : KERNEL_NAME_CUBES);
				this->err_str+='\"';
				return false;
			}
			break;
		case SPHERES:
			kernel=clCreateKernel(program,on? KERNEL_NAME_SPHERES_VOL : KERNEL_NAME_SPHERES,NULL);
			if (kernel==NULL)
			{
				this->err_str=QObject::tr("no such kernel named \"");
				this->err_str+=(on? KERNEL_NAME_SPHERES_VOL : KERNEL_NAME_SPHERES);
				this->err_str+='\"';
				return false;
			}
			break;
		case LIGHTS:
			kernel=clCreateKernel(program,on? KERNEL_NAME_LIGHTS_VOL : KERNEL_NAME_LIGHTS,NULL);
			if (kernel==NULL)
			{
				this->err_str=QObject::tr("no such kernel named \"");
				this->err_str+=(on? KERNEL_NAME_LIGHTS_VOL : KERNEL_NAME_LIGHTS);
				this->err_str+='\"';
				return false;
			}
			break;
		default: break; // impossible!
	}
	this->err_str.clear();
	if (clSetKernelArg(kernel,0,sizeof(cl_float16),&data)!=CL_SUCCESS)
		ARG_PASS_ERR(0,false)
	if (clSetKernelArg(kernel,1,sizeof(cl_int3),&num)!=CL_SUCCESS)
		ARG_PASS_ERR(1,false)
	if (on)
	{
		if (clSetKernelArg(kernel,2,sizeof(cl_uchar4),&clr_minmax)!=CL_SUCCESS)
			ARG_PASS_ERR(2,pt_repr)
	}
	else
	{
		if (clSetKernelArg(kernel,2,sizeof(int),&depth_constraint)!=CL_SUCCESS)
			ARG_PASS_ERR(2,false)
	}
	if (clSetKernelArg(kernel,3,sizeof(cl_mem),&matrix)!=CL_SUCCESS)
		ARG_PASS_ERR(3,false)
	if (clSetKernelArg(kernel,4,sizeof(gpu_pixels),&gpu_pixels)!=CL_SUCCESS)
		ARG_PASS_ERR(4,false)
	volume_mode=on;
	return true;
}

bool RendererOCL::BuildVolume (const unsigned char min_green, const unsigned char max_green, 
							   const unsigned char min_red, const unsigned char max_red) {
	this->err_str.clear();
	clr_minmax.x=min_green;
	clr_minmax.y=max_green;
	clr_minmax.z=min_red;
	clr_minmax.w=max_red;
	if (clSetKernelArg(kernel,2,sizeof(cl_uchar4),&clr_minmax)!=CL_SUCCESS)
		ARG_PASS_ERR(2,false)
	return true;
}

void RendererOCL::SelectPoints_by_click (const int x, const int y, const unsigned short *const clr_matrix, 
										 Coords* &pos, int &points_num) const {
	/* see original algorithm in "./renderer_OpenCL.ocl" */

	pos=NULL;
	points_num=0;

	const float point_step=data.sc;
	const float h_sizes_x=((float)num.x)*point_step*0.5f;
	const float h_sizes_y=((float)num.y)*point_step*0.5f;
	const float h_sizes_z=((float)num.z)*point_step*0.5f;
	const float ray_pos_const_x=data.s9*data.s0+data.sa*data.s1+data.sb*data.s2;
	const float ray_pos_const_y=data.s9*data.s3+data.sa*data.s4+data.sb*data.s5;
	const float ray_pos_const_z=data.s9*data.s6+data.sa*data.s7+data.sb*data.s8;
	const float trans_x=static_cast<float>(x+x-static_cast<int>(img_w));
	const float trans_y=static_cast<float>(y+y-static_cast<int>(img_h));
	float trans_z=(1.0f+M_SQRT2)*static_cast<float>(img_w);
	float ray_dir_x=trans_x*data.s0+trans_y*data.s1+trans_z*data.s2;
	float ray_dir_y=trans_x*data.s3+trans_y*data.s4+trans_z*data.s5;
	float ray_dir_z=trans_x*data.s6+trans_y*data.s7+trans_z*data.s8;

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

	const float p_st_div_r_d_x=point_step/ray_dir_x,p_st_div_r_d_y=point_step/ray_dir_y,p_st_div_r_d_z=point_step/ray_dir_z;
	int ind_x=(r_p_div_p_st_x<1.0f)? 0 : ((r_p_div_p_st_x>=(float)(num.x-1))? (num.x-1) : (int)floor(r_p_div_p_st_x));
	int ind_y=(r_p_div_p_st_y<1.0f)? 0 : ((r_p_div_p_st_y>=(float)(num.y-1))? (num.y-1) : (int)floor(r_p_div_p_st_y));
	int ind_z=(r_p_div_p_st_z<1.0f)? 0 : ((r_p_div_p_st_z>=(float)(num.z-1))? (num.z-1) : (int)floor(r_p_div_p_st_z));
	const int ind_step_x=(ray_dir_x<0.0f)? -1 : 1;
	const int ind_step_y=(ray_dir_y<0.0f)? -1 : 1;
	const int ind_step_z=(ray_dir_z<0.0f)? -1 : 1;
	float t_max_x=p_st_div_r_d_x*((float)(ind_x+((ind_step_x+1)>>1u))-r_p_div_p_st_x);
	float t_max_y=p_st_div_r_d_y*((float)(ind_y+((ind_step_y+1)>>1u))-r_p_div_p_st_y);
	float t_max_z=p_st_div_r_d_z*((float)(ind_z+((ind_step_z+1)>>1u))-r_p_div_p_st_z);
	const float t_delta_x=p_st_div_r_d_x*(float)ind_step_x;
	const float t_delta_y=p_st_div_r_d_y*(float)ind_step_y;
	const float t_delta_z=p_st_div_r_d_z*(float)ind_step_z;
	const int xy_num=num.x*num.y*ind_step_z;
	const int yl_num=num.x*ind_step_y;
	int mtr_indx=(ind_z*num.y+ind_y)*num.x+ind_x;

	if (volume_mode)
	{
		// 'volume' mode is on

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
					if ((ind_x>=num.x) || (ind_x<0)) break;
					t_max_x+=t_delta_x;
					mtr_indx+=ind_step_x;
				}
				else
				{
					ind_z+=ind_step_z;
					if ((ind_z>=num.z) || (ind_z<0)) break;
					t_max_z+=t_delta_z;
					mtr_indx+=xy_num;
				}
			}
			else
			{
				if (t_max_y<=t_max_z)
				{
					ind_y+=ind_step_y;
					if ((ind_y>=num.y) || (ind_y<0)) break;
					t_max_y+=t_delta_y;
					mtr_indx+=yl_num;
				}
				else
				{
					ind_z+=ind_step_z;
					if ((ind_z>=num.z) || (ind_z<0)) break;
					t_max_z+=t_delta_z;
					mtr_indx+=xy_num;
				}
			}
		}
	}
	else
	{
		// 'volume' mode is off

		const unsigned int ind_init_x=ind_x,ind_init_y=ind_y,ind_init_z=ind_z;

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
					if ((ind_x>=num.x) || (ind_x<0) || (ind_init_x>ind_x+depth_constraint) || 
						(ind_init_x+depth_constraint<static_cast<unsigned>(ind_x))) break;
					t_max_x+=t_delta_x;
					mtr_indx+=ind_step_x;
				}
				else
				{
					ind_z+=ind_step_z;
					if ((ind_z>=num.z) || (ind_z<0) || (ind_init_z>ind_z+depth_constraint) || 
						(ind_init_z+depth_constraint<static_cast<unsigned>(ind_z))) break;
					t_max_z+=t_delta_z;
					mtr_indx+=xy_num;
				}
			}
			else
			{
				if (t_max_y<=t_max_z)
				{
					ind_y+=ind_step_y;
					if ((ind_y>=num.y) || (ind_y<0) || (ind_init_y>ind_y+depth_constraint) || 
						(ind_init_y+depth_constraint<static_cast<unsigned>(ind_y))) break;
					t_max_y+=t_delta_y;
					mtr_indx+=yl_num;
				}
				else
				{
					ind_z+=ind_step_z;
					if ((ind_z>=num.z) || (ind_z<0) || (ind_init_z>ind_z+depth_constraint) || 
						(ind_init_z+depth_constraint<static_cast<unsigned>(ind_z))) break;
					t_max_z+=t_delta_z;
					mtr_indx+=xy_num;
				}
			}
		}
	}
}

bool RendererOCL::SelectPoints_last_phase (const int x, const int y, Coords &pos, 
										   const unsigned short *const clr_matrix) {
	/* see original algorithm in "./renderer_OpenCL.ocl" */

	const float point_step=data.sc;
	const float h_sizes_x=((float)num.x)*point_step*0.5f;
	const float h_sizes_y=((float)num.y)*point_step*0.5f;
	const float h_sizes_z=((float)num.z)*point_step*0.5f;
	const float ray_pos_const_x=data.s9*data.s0+data.sa*data.s1+data.sb*data.s2;
	const float ray_pos_const_y=data.s9*data.s3+data.sa*data.s4+data.sb*data.s5;
	const float ray_pos_const_z=data.s9*data.s6+data.sa*data.s7+data.sb*data.s8;
	const float trans_x=(float)(x+x-(int)img_w);
	const float trans_y=(float)(y+y-(int)img_h);
	float trans_z=(1.0f+M_SQRT2)*(float)img_w;
	float ray_dir_x=trans_x*data.s0+trans_y*data.s1+trans_z*data.s2;
	float ray_dir_y=trans_x*data.s3+trans_y*data.s4+trans_z*data.s5;
	float ray_dir_z=trans_x*data.s6+trans_y*data.s7+trans_z*data.s8;

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

	const float p_st_div_r_d_x=point_step/ray_dir_x,p_st_div_r_d_y=point_step/ray_dir_y,p_st_div_r_d_z=point_step/ray_dir_z;
	int ind_x=(r_p_div_p_st_x<1.0f)? 0 : ((r_p_div_p_st_x>=(float)(num.x-1))? (num.x-1) : (int)floor(r_p_div_p_st_x));
	int ind_y=(r_p_div_p_st_y<1.0f)? 0 : ((r_p_div_p_st_y>=(float)(num.y-1))? (num.y-1) : (int)floor(r_p_div_p_st_y));
	int ind_z=(r_p_div_p_st_z<1.0f)? 0 : ((r_p_div_p_st_z>=(float)(num.z-1))? (num.z-1) : (int)floor(r_p_div_p_st_z));
	const int ind_step_x=(ray_dir_x<0.0f)? -1 : 1;
	const int ind_step_y=(ray_dir_y<0.0f)? -1 : 1;
	const int ind_step_z=(ray_dir_z<0.0f)? -1 : 1;
	float t_max_x=p_st_div_r_d_x*((float)(ind_x+((ind_step_x+1)>>1))-r_p_div_p_st_x);
	float t_max_y=p_st_div_r_d_y*((float)(ind_y+((ind_step_y+1)>>1))-r_p_div_p_st_y);
	float t_max_z=p_st_div_r_d_z*((float)(ind_z+((ind_step_z+1)>>1))-r_p_div_p_st_z);
	const float t_delta_x=p_st_div_r_d_x*(float)ind_step_x;
	const float t_delta_y=p_st_div_r_d_y*(float)ind_step_y;
	const float t_delta_z=p_st_div_r_d_z*(float)ind_step_z;
	const int xy_num=num.x*num.y*ind_step_z;
	const int yl_num=num.x*ind_step_y;
	int mtr_indx=(ind_z*num.y+ind_y)*num.x+ind_x;
	float to_centre_x,to_centre_y,to_centre_z;
	const float point_step_sq=point_step*point_step;
	bool was_hit;

	for ( ; ; )
	{
		if (pt_repr==CUBES) was_hit=true;
		else
		{
			to_centre_x=((float)ind_x+0.5f-r_p_div_p_st_x);
			to_centre_y=((float)ind_y+0.5f-r_p_div_p_st_y);
			to_centre_z=((float)ind_z+0.5f-r_p_div_p_st_z);
			trans_z=to_centre_x*ray_dir_x+to_centre_y*ray_dir_y+to_centre_z*ray_dir_z;
			was_hit=(data.sd>=point_step_sq*((to_centre_x-trans_z)*(to_centre_x+trans_z)+
					 to_centre_y*to_centre_y+to_centre_z*to_centre_z));
		}
		if (was_hit && (clr_matrix[mtr_indx]!=0x00))
		{
			pos.x=ind_x;
			pos.y=ind_y;
			pos.z=ind_z;

			pt_selection_mode=true;
			sel_ind.x=ind_x;
			sel_ind.y=ind_y;
			sel_ind.z=ind_z;
			BuildSelectionCube(ray_pos_const_x+h_sizes_x,ray_pos_const_y+h_sizes_y,ray_pos_const_z+h_sizes_z);
			return true;
		}
		if (t_max_x<=t_max_y)
		{
			if (t_max_x<=t_max_z)
			{
				ind_x+=ind_step_x;
				if ((ind_x>=num.x) || (ind_x<0)) break;
				t_max_x+=t_delta_x;
				mtr_indx+=ind_step_x;
			}
			else
			{
				ind_z+=ind_step_z;
				if ((ind_z>=num.z) || (ind_z<0)) break;
				t_max_z+=t_delta_z;
				mtr_indx+=xy_num;
			}
		}
		else
		{
			if (t_max_y<=t_max_z)
			{
				ind_y+=ind_step_y;
				if ((ind_y>=num.y) || (ind_y<0)) break;
				t_max_y+=t_delta_y;
				mtr_indx+=yl_num;
			}
			else
			{
				ind_z+=ind_step_z;
				if ((ind_z>=num.z) || (ind_z<0)) break;
				t_max_z+=t_delta_z;
				mtr_indx+=xy_num;
			}
		}
	}
	return false;
}

void RendererOCL::BuildSelectionCube (const float ray_pos_x, const float ray_pos_y, const float ray_pos_z) const {
	/* don't forget that all calculations are carried in bounding box coordinate system! */

	memset(fv_sel_cube_vis,0,8u*sizeof(bool));

	float to_centre_x=data.sc*(float)(sel_ind.x+2)-ray_pos_x;
	float to_centre_y=data.sc*(float)(sel_ind.y+2)-ray_pos_y;
	float to_centre_z=data.sc*(float)(sel_ind.z+2)-ray_pos_z;

	/* determine visibility of 8 corner points of selection cube (5 big steps below) */

	if ((to_centre_x<=data.sc+data.sc) && (data.sc<=to_centre_x) && 
		(to_centre_y<=data.sc+data.sc) && (data.sc<=to_centre_y) && 
		(to_centre_z<=data.sc+data.sc) && (data.sc<=to_centre_z))
		// 'ray_pos' is inside the cube - assume the whole cube invisible
		return;

	const float trans_z=(1.0f+M_SQRT2)*(const float)img_w;
	const float img_w_2=0.5f*(const float)img_w,img_h_2=0.5f*(const float)img_h;
	float a,b,c,d,e,f=-1.5f*data.sc,mlt;
	bool hit=false;

	/* STEP 1: transform centre of selected "point" into screen coordinates 
	           ('d' will be (j-img_w/2), 'e' will be (i-img_h/2)) */
	a=(to_centre_x+f)*data.s7-(to_centre_z+f)*data.s1;
	b=(to_centre_x+f)*data.s3-(to_centre_y+f)*data.s0;
	c=(to_centre_x+f)*data.s6-(to_centre_z+f)*data.s0;
	d=(to_centre_x+f)*data.s4-(to_centre_y+f)*data.s1;
	mlt=a*b-c*d;
	if ((mlt<FLT_EPSILON) && (mlt>-FLT_EPSILON))
	{
		a=(to_centre_y+f)*data.s7-(to_centre_z+f)*data.s4;
		c=(to_centre_y+f)*data.s6-(to_centre_z+f)*data.s3;
		mlt=a*b-c*d;
		if ((mlt<FLT_EPSILON) && (mlt>-FLT_EPSILON))
		{
			mlt=trans_z*data.s2/data.s8;
			d=floor(img_w_2+(data.s4-data.s1)*mlt+0.5f)-img_w_2;
			e=floor(img_h_2+(data.s0-data.s3)*mlt+0.5f)-img_h_2;
		}
		else
		{
			e=(to_centre_y+f)*data.s8-(to_centre_z+f)*data.s5;
			f=(to_centre_x+f)*data.s5-(to_centre_y+f)*data.s2;
			mlt=trans_z/(mlt+mlt);
			d=floor((e*d-a*f)*mlt+img_w_2+0.5f)-img_w_2;
			e=floor((c*f-e*b)*mlt+img_h_2+0.5f)-img_h_2;
		}
	}
	else
	{
		e=(to_centre_x+f)*data.s8-(to_centre_z+f)*data.s2;
		f=(to_centre_x+f)*data.s5-(to_centre_y+f)*data.s2;
		mlt=trans_z/(mlt+mlt);
		d=floor((e*d-a*f)*mlt+img_w_2+0.5f)-img_w_2;
		e=floor((c*f-e*b)*mlt+img_h_2+0.5f)-img_h_2;
	}
	if ((d<=-img_w_2) || (d>=img_w_2) || (e<=-img_h_2) || (e>=img_h_2))
		// centre of selected "point" is invisible - assume the whole cube invisible
		return;
	/* compute a ray from 'ray_pos' to found coordinates */
	f=0.5f*trans_z;
	a=d*data.s0+e*data.s1+f*data.s2;
	b=d*data.s3+e*data.s4+f*data.s5;
	c=d*data.s6+e*data.s7+f*data.s8;
	// no need in normalization of 'a', 'b' and 'c' (aka 'ray_dir')

	/* STEP 2: find intersection of the ray and faces of selected "point" */
	if ((c>FLT_EPSILON) && (data.sc+data.sc<=to_centre_z))
	{
		e=(to_centre_z-data.sc-data.sc)/c;
		mlt=a*e+data.sc;
		if ((mlt+data.sc>=to_centre_x) && (mlt<=to_centre_x))
		{
			mlt=b*e+data.sc;
			if ((mlt+data.sc>=to_centre_y) && (mlt<=to_centre_y)) hit=true;
		}
	}
	else
		if ((c<-FLT_EPSILON) && (data.sc>=to_centre_z))
		{
			e=(to_centre_z-data.sc)/c;
			mlt=a*e+data.sc;
			if ((mlt+data.sc>=to_centre_x) && (mlt<=to_centre_x))
			{
				mlt=b*e+data.sc;
				if ((mlt+data.sc>=to_centre_y) && (mlt<=to_centre_y)) hit=true;
			}
		}
	if (!hit)
	{
		if ((b>FLT_EPSILON) && (data.sc+data.sc<=to_centre_y))
		{
			e=(to_centre_y-data.sc-data.sc)/b;
			mlt=a*e+data.sc;
			if ((mlt+data.sc>=to_centre_x) && (mlt<=to_centre_x))
			{
				mlt=c*e+data.sc;
				if ((mlt+data.sc>=to_centre_z) && (mlt<=to_centre_z)) hit=true;
			}
		}
		else
			if ((b<-FLT_EPSILON) && (data.sc>=to_centre_y))
			{
				e=(to_centre_y-data.sc)/b;
				mlt=a*e+data.sc;
				if ((mlt+data.sc>=to_centre_x) && (mlt<=to_centre_x))
				{
					mlt=c*e+data.sc;
					if ((mlt+data.sc>=to_centre_z) && (mlt<=to_centre_z)) hit=true;
				}
			}
		if (!hit)
		{
			if ((a>FLT_EPSILON) && (data.sc+data.sc<=to_centre_x))
			{
				e=(to_centre_x-data.sc-data.sc)/a;
				mlt=b*e+data.sc;
				if ((mlt+data.sc>=to_centre_y) && (mlt<=to_centre_y))
				{
					mlt=c*e+data.sc;
					if ((mlt+data.sc>=to_centre_z) && (mlt<=to_centre_z)) hit=true;
				}
			}
			else
				if ((a<-FLT_EPSILON) && (data.sc>=to_centre_x))
				{
					e=(to_centre_x-data.sc)/a;
					mlt=b*e+data.sc;
					if ((mlt+data.sc>=to_centre_y) && (mlt<=to_centre_y))
					{
						mlt=c*e+data.sc;
						if ((mlt+data.sc>=to_centre_z) && (mlt<=to_centre_z)) hit=true;
					}
				}
		}
	}
	if (!hit) return; // no intersection - the selected "point" itself is invisible

	char i,j,k;
	int ind=0;

	/* STEP 3: find the nearest point of the cube to 'ray_pos' */
	a=FLT_MAX;
	to_centre_z-=(data.sc+data.sc);
	for (k=0; k<8; k+=4)
	{
		d=to_centre_z*to_centre_z;
		to_centre_y-=(data.sc+data.sc);
		for (j=0; j<2; ++j)
		{
			c=to_centre_y*to_centre_y+d;
			to_centre_x-=(data.sc+data.sc);
			for (i=0; i<2; ++i)
			{
				b=to_centre_x*to_centre_x+c;
				if (b<a)
				{
					a=b;
					ind=k+j+j+i;
				}
				to_centre_x+=data.sc;
			}
			to_centre_y+=data.sc;
		}
		to_centre_z+=data.sc;
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
	f=-1.5f*data.sc;
	for (ind=0; ind<3; ++ind)
	{
		switch (faces[ind])
		{
			case 0:
				a=to_centre_x+f;
				b=to_centre_y+f;
				c=to_centre_z-data.sc-data.sc;
				break;
			case 1:
				a=to_centre_x+f;
				b=to_centre_y-data.sc-data.sc;
				c=to_centre_z+f;
				break;
			case 2:
				a=to_centre_x-data.sc-data.sc;
				b=to_centre_y+f;
				c=to_centre_z+f;
				break;
			case 3:
				a=to_centre_x+f;
				b=to_centre_y+f;
				c=to_centre_z-data.sc;
				break;
			case 4:
				a=to_centre_x+f;
				b=to_centre_y-data.sc;
				c=to_centre_z+f;
				break;
			case 5:
				a=to_centre_x-data.sc;
				b=to_centre_y+f;
				c=to_centre_z+f;
				break;
		}
		for (j=1; j<3; ++j)
		{
			i=faces[(ind+(int)j)%3];
			mlt=-data.sc;
			if (i<3) mlt-=data.sc;
			switch (i)
			{
				case 0:
				case 3:
					if ((c<-FLT_EPSILON) || (c>FLT_EPSILON))
					{
						mlt=(to_centre_z+mlt)/c;
						if (mlt>0.0f)
						{
							d=mlt*a+data.sc;
							if ((d+data.sc>to_centre_x) && (d<to_centre_x))
							{
								d=mlt*b+data.sc;
								if ((d+data.sc>to_centre_y) && (d<to_centre_y)) vis[ind]=false;
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
							d=mlt*a+data.sc;
							if ((d+data.sc>to_centre_x) && (d<to_centre_x))
							{
								d=mlt*c+data.sc;
								if ((d+data.sc>to_centre_z) && (d<to_centre_z)) vis[ind]=false;
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
							d=mlt*b+data.sc;
							if ((d+data.sc>to_centre_y) && (d<to_centre_y))
							{
								d=mlt*c+data.sc;
								if ((d+data.sc>to_centre_z) && (d<to_centre_z)) vis[ind]=false;
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
	to_centre_z-=(data.sc+data.sc);
	for (k=0; k<2; ++k)
	{
		to_centre_y-=(data.sc+data.sc);
		for (j=0; j<2; ++j)
		{
			to_centre_x-=(data.sc+data.sc);
			for (i=0; i<2; ++i)
			{
				if (fv_sel_cube_vis[ind])
				{
					a=to_centre_x*data.s7-to_centre_z*data.s1;
					b=to_centre_x*data.s3-to_centre_y*data.s0;
					c=to_centre_x*data.s6-to_centre_z*data.s0;
					d=to_centre_x*data.s4-to_centre_y*data.s1;
					mlt=a*b-c*d;
					if ((mlt<FLT_EPSILON) && (mlt>-FLT_EPSILON))
					{
						a=to_centre_y*data.s7-to_centre_z*data.s4;
						c=to_centre_y*data.s6-to_centre_z*data.s3;
						mlt=a*b-c*d;
						if ((mlt<FLT_EPSILON) && (mlt>-FLT_EPSILON))
						{
							mlt=trans_z*data.s2/data.s8;
							fv_sel_cube_x[ind]=(int)floor(img_w_2+(data.s4-data.s1)*mlt+0.5f);
							fv_sel_cube_y[ind]=(int)floor(img_h_2+(data.s0-data.s3)*mlt+0.5f);
							to_centre_x+=data.sc;
							++ind;
							continue;
						}
						e=to_centre_y*data.s8-to_centre_z*data.s5;
					}
					else e=to_centre_x*data.s8-to_centre_z*data.s2;
					f=to_centre_x*data.s5-to_centre_y*data.s2;
					mlt=trans_z/(mlt+mlt);
					fv_sel_cube_x[ind]=(int)floor((e*d-a*f)*mlt+img_w_2+0.5f);
					fv_sel_cube_y[ind]=(int)floor((c*f-e*b)*mlt+img_h_2+0.5f);
				}
				to_centre_x+=data.sc;
				++ind;
			}
			to_centre_y+=data.sc;
		}
		to_centre_z+=data.sc;
	}
}

RendererOCL::~RendererOCL () {
	if (comm_queue!=NULL)
	{
		clFinish(comm_queue);
		clReleaseCommandQueue(comm_queue);
	}
	if (gpu_pixels!=NULL) clReleaseMemObject(gpu_pixels);
	if (matrix!=NULL) clReleaseMemObject(matrix);
	if (kernel!=NULL) clReleaseKernel(kernel);
	if (program!=NULL) clReleaseProgram(program);
	if (contxt!=NULL) clReleaseContext(contxt);
}

