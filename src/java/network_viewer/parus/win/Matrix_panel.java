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


import javax.swing.JFrame;
import javax.swing.JPanel;
import java.awt.Container;

import java.awt.image.BufferedImage;

import java.awt.Graphics2D;
import java.awt.Graphics;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.Dimension;
import java.awt.Color;


import parus.data.Matrix;


public class Matrix_panel extends JPanel
{
	private Matrix mtr;
	private Matrix deviation_mtr;
	private BufferedImage img;

	private double x_resize_coef;
	private double y_resize_coef;

	private int mtr_size_x;
	private int mtr_size_y;

	private int pointer_x_coord=0;
	private int pointer_y_coord=0;

	private int pointer_x_coord2=0;
	private int pointer_y_coord2=0;

	private double min;
	private double max;

	public Matrix_panel(Matrix m,Matrix deviation,boolean normolize)
	{
		super();
		
		mtr_size_x=m.get_size_x();
		mtr_size_y=m.get_size_y();
		max=m.get_max_element();
		min=m.get_min_element();
		
		this.update(m,deviation);
		
	}

	public Matrix_panel(Matrix m,Matrix deviation,boolean normolize, int fromx, int fromy, int tox, int toy)
	{
		super();

		mtr_size_x=(fromx>=tox?fromx-tox:tox-fromx)+1;
		mtr_size_y=(fromy>=toy?fromy-toy:toy-fromy)+1;

		max=m.get_element(fromx,fromy);
		min=max;
		for(int i=(fromx<=tox?fromx:tox);i<=(fromx>=tox?fromx:tox);i++)
			for(int j=(fromy<=toy?fromy:toy);j<=(fromy>=toy?fromy:toy);j++)
			{
				if(max<m.get_element(i,j)) max=m.get_element(i,j);
				if(min>m.get_element(i,j)) min=m.get_element(i,j);
			}
		this.update(m,deviation,fromx,fromy,tox,toy);
		
	}

	public void paint(Graphics g)
	{
		super.paint(g);

		Graphics2D graphics=(Graphics2D)g;
		Dimension dimension;

		dimension=this.getSize(null);
				
		x_resize_coef=(double)(dimension.width)/mtr_size_x;
		y_resize_coef=(double)(dimension.height)/mtr_size_y;
		
		AffineTransform resizer= new AffineTransform
		(
			x_resize_coef, 0.0, 0.0,
			y_resize_coef, 0.0, 0.0
		);
		
		AffineTransformOp op = new AffineTransformOp
		(
			resizer,
			//AffineTransformOp.TYPE_BICUBIC
			AffineTransformOp.TYPE_NEAREST_NEIGHBOR
		);
		
		
		graphics.drawImage(img,op,0,0);

		graphics.setColor(Color.GREEN);
		
		int x_center_point_coord;
		int y_center_point_coord;
		int x_center_point_coord2;
		int y_center_point_coord2;
		if(x_resize_coef > 1)
		{
			x_center_point_coord=(int)(pointer_x_coord*x_resize_coef+x_resize_coef*0.5);	
			x_center_point_coord2=(int)(pointer_x_coord2*x_resize_coef+x_resize_coef*0.5);
		}
		else
		{
			x_center_point_coord=(int)(pointer_x_coord*x_resize_coef);
			x_center_point_coord2=(int)(pointer_x_coord2*x_resize_coef);
		}

		if(y_resize_coef > 1)
		{
			y_center_point_coord=(int)(pointer_y_coord*y_resize_coef+y_resize_coef*0.5);
			y_center_point_coord2=(int)(pointer_y_coord2*y_resize_coef+y_resize_coef*0.5);
		}
		else
		{
			y_center_point_coord=(int)(pointer_y_coord*y_resize_coef);
			y_center_point_coord2=(int)(pointer_y_coord2*y_resize_coef);
		}
		
		graphics.drawLine
		(
			x_center_point_coord-20,
			y_center_point_coord,
			x_center_point_coord+20,
			y_center_point_coord
		);
	
		graphics.drawLine
		(
			x_center_point_coord,
			y_center_point_coord-20,
			x_center_point_coord,
			y_center_point_coord+20
		);
		graphics.setColor(Color.BLUE);
		/*graphics.drawLine
		(
			x_center_point_coord2-20,
			y_center_point_coord2,
			x_center_point_coord2+20,
			y_center_point_coord2
		);

		graphics.drawLine
		(
			x_center_point_coord2,
			y_center_point_coord2-20,
			x_center_point_coord2,
			y_center_point_coord2+20
		);*/
		graphics.drawLine
		(
			x_center_point_coord,
			y_center_point_coord,
			x_center_point_coord2,
			y_center_point_coord
		);
		graphics.drawLine
		(
			x_center_point_coord,
			y_center_point_coord,
			x_center_point_coord,
			y_center_point_coord2
		);
		graphics.drawLine
		(
			x_center_point_coord2,
			y_center_point_coord2,
			x_center_point_coord,
			y_center_point_coord2
		);
		graphics.drawLine
		(
			x_center_point_coord2,
			y_center_point_coord2,
			x_center_point_coord2,
			y_center_point_coord
		);
		
	}

	public double get_x_resize_coef()
	{
		return x_resize_coef;
	}

	public double get_y_resize_coef()
	{
		return y_resize_coef;
	}

	public void set_pointer_x_coord(int pointer)
	{
		pointer_x_coord=pointer;
	}
	
	public void set_pointer_y_coord(int pointer)
	{
		pointer_y_coord=pointer;
	}

	public int get_pointer_x_coord()
	{
		return pointer_x_coord;
	}
	
	public int get_pointer_y_coord()
	{
		return pointer_y_coord;
	}

	public void set_pointer_x_coord2(int pointer)
	{
		pointer_x_coord2=pointer;
	}
	
	public void set_pointer_y_coord2(int pointer)
	{
		pointer_y_coord2=pointer;
	}

	public int get_pointer_x_coord2()
	{
		return pointer_x_coord2;
	}
	
	public int get_pointer_y_coord2()
	{
		return pointer_y_coord2;
	}

	public void set_min_matrix_value(double val)
	{
		min=val;
	}

	public void set_max_matrix_value(double val)
	{
		max=val;
	}

	public double get_min_matrix_value()
	{
		return min;
	}

	public double get_max_matrix_value()
	{
		return max;
	}

	public void update(Matrix m,Matrix deviation)
	{
		mtr=m;
		deviation_mtr=deviation;
		Matrix_to_image converter;
		if(deviation_mtr==null)
		{
			converter=new Matrix_to_image(mtr,min,max);
		}
		else
		{
			converter=new Matrix_to_image(mtr,deviation_mtr,min,max);
		}
		img=converter.get_image();
	}
	
	public void update(Matrix m,Matrix deviation, int fromx, int fromy, int tox, int toy)
	{
		mtr=m;
		deviation_mtr=deviation;
		
		Matrix_to_image converter;

		if(deviation_mtr==null)
		{
			converter=new Matrix_to_image
							(
								mtr,min,max,
								fromx>=tox?tox:fromx,
								fromy>=toy?toy:fromy,
								fromx<=tox?tox:fromx,
								fromy<=toy?toy:fromy
							);
		}
		else
		{
			converter=new Matrix_to_image
							(
								mtr,deviation_mtr,min,max,
								fromx>=tox?tox:fromx,
								fromy>=toy?toy:fromy,
								fromx<=tox?tox:fromx,
								fromy<=toy?toy:fromy
							);

		}
		img=converter.get_image();
	}

	public static void main(String args[])
	{
		Matrix m=new Matrix(1000,1000);

		for(int i=0;i < 1000; i++)
		for(int j=0; j< 1000; j++)
		{
			m.set_element(i,j,(double)(i+j));
		}
		
		JFrame shower = new JFrame("Matrix panel");
		Container pane=shower.getContentPane();
		shower.setSize(300,300);
		pane.add(new Matrix_panel(m,null,true));
		shower.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		shower.setVisible(true);
	}

	

}

