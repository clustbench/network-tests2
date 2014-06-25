/*
 *  This file is a part of the PARUS project.
 *  Copyright (C) 2006  Alexey N. Salnikov
 *  
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 * 
 * Alexey N. Salnikov salnikov@cmc.msu.ru
 *
 */


package parus.win;

import parus.data.Matrix;

import java.awt.image.BufferedImage;

public class Matrix_to_image
{
	BufferedImage img;

	public Matrix_to_image(Matrix mtr,double min,double max)
	{
		int sx=mtr.get_size_x();
		int sy=mtr.get_size_y();
		
		/*
		double max=mtr.get_max_element();
		double min=mtr.get_min_element();
		*/

		img=new BufferedImage(sx,sy,BufferedImage.TYPE_INT_RGB);

		double current_element;
		int pixel_component=255;
		int pixel;

		for(int i=0;i<sx;i++)
		for(int j=0;j<sy;j++)
		{
			current_element=mtr.get_element(i,j);
			
			if(current_element<min)
			{
				current_element=min;
			}

			if(current_element>max)
			{
				current_element=max;
			}
			if(max>min)
			{
				pixel_component=(int)(255.0-((double)(current_element-min)/(double)(max-min))*255.0);
			}
			pixel=pixel_component*0x010101;
			img.setRGB(i,j,pixel);
		}
	}
	
	public Matrix_to_image(Matrix mtr,Matrix deviation_mtr, double min, double max)
	{
		int sx=mtr.get_size_x();
		int sy=mtr.get_size_y();
		
		/*
		double max=mtr.get_max_element();
		double min=mtr.get_min_element();
		*/

		img=new BufferedImage(sx,sy,BufferedImage.TYPE_INT_RGB);

		double current_element;
		int pixel_component=255;
		int pixel;
		
		double current_deviation;
		double deviation_coefficient=1.0;
		
		for(int i=0;i<sx;i++)
		for(int j=0;j<sy;j++)
		{
			current_deviation = deviation_mtr.get_element(i,j);
			current_element   = mtr.get_element(i,j);
				
			if(current_element<min)
			{
				current_element=min;
			}

			if(current_element>max)
			{
				current_element=max;
			}


			if(current_deviation>=(current_element))
			{
				deviation_coefficient=0.0;
			}
			else
			{
				deviation_coefficient=1.0-current_deviation/(current_element);
			}

			if(current_deviation==0.0)
			{
				deviation_coefficient=1.0;
			}
			
			if(max>min)
			{
				pixel_component=(int)(255.0-((double)(current_element-min)/(double)(max-min))*255.0);
			}
			pixel=(int)(pixel_component*deviation_coefficient)*0x000101;
			pixel+=pixel_component*0x010000;

			img.setRGB(i,j,pixel);
		}
	}

	public Matrix_to_image(Matrix mtr,double min,double max, int fromx, int fromy, int tox, int toy)
	{
		/*
		int sx=mtr.get_size_x();
		int sy=mtr.get_size_y();
		*/
		int sx=tox-fromx+1;
		int sy=toy-fromy+1;
		
		/*
		double max=mtr.get_max_element();
		double min=mtr.get_min_element();
		*/

		img=new BufferedImage(tox-fromx+1,toy-fromy+1,BufferedImage.TYPE_INT_RGB);

		double current_element;
		int pixel_component=255;
		int pixel;

		for(int i=fromx;i<=tox;i++)
			for(int j=fromy;j<=toy;j++)
			{
				current_element=mtr.get_element(i,j);
					
				if(current_element<min)
				{
					current_element=min;
				}

				if(current_element>max)
				{
					current_element=max;
				}
				if(max>min)
				{
					pixel_component=(int)(255.0-((double)(current_element-min)/(double)(max-min))*255.0);
				}
				pixel=pixel_component*0x010101;
				img.setRGB(i-fromx,j-fromy,pixel);
			}
	}
	
	public Matrix_to_image(Matrix mtr, Matrix deviation_mtr, double min,double max, int fromx, int fromy, int tox, int toy)
	{
		/*
		int sx=mtr.get_size_x();
		int sy=mtr.get_size_y();
		*/
		int sx=tox-fromx+1;
		int sy=toy-fromy+1;
		
		/*
		double max=mtr.get_max_element();
		double min=mtr.get_min_element();
		*/
		
		img=new BufferedImage(tox-fromx+1,toy-fromy+1,BufferedImage.TYPE_INT_RGB);

		double current_element;
		double current_deviation;
		int pixel_component=255;
		double deviation_coefficient=1.0;
		int pixel;

		for(int i=fromx;i<=tox;i++)
			for(int j=fromy;j<=toy;j++)
			{
				current_element=mtr.get_element(i,j);
				
				if(current_element<min)
				{
					current_element=min;
				}

				if(current_element>max)
				{
					current_element=max;
				}
				
				current_deviation=deviation_mtr.get_element(i,j);
				if(current_deviation>=(current_element))
				{
					deviation_coefficient=0.0;
				}
				else
				{
					deviation_coefficient=1.0-current_deviation/(current_element);			
				}

				if(current_deviation==0)
				{
					deviation_coefficient=1.0;
				}
				
				if(max>min)
				{
					pixel_component=(int)(255.0-((double)(current_element-min)/(double)(max-min))*255.0);
				}

				pixel=(int)(pixel_component*deviation_coefficient)*0x000101;
				pixel+=pixel_component*0x010000;	
				img.setRGB(i-fromx,j-fromy,pixel);
			}
	}


	public BufferedImage get_image()
	{
		return img;
	}
}
