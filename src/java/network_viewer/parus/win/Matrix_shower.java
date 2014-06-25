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
import parus.common.Objects_loader;

import java.net.URL;

import java.awt.Dialog;
import javax.swing.JDialog;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.JButton;

import javax.swing.JSlider;
import javax.swing.event.ChangeListener;
import javax.swing.event.ChangeEvent;

import javax.swing.ImageIcon;

import java.awt.Container;

import java.awt.image.BufferedImage;

import java.awt.Graphics2D;
import java.awt.Graphics;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.Dimension;

import java.awt.BorderLayout;
import java.awt.GridLayout;

import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;




public class Matrix_shower extends JDialog
{
	private Matrix mtr;
	private Matrix deviation;
	private Matrix_panel panel;
	
	private JTextField from,from2;
	private JTextField to,to2;
	private JTextField current_value;
	private JTextField current_value2;
	private JDialog tframe;

	private JSlider white_slider;
	private JSlider black_slider;

	private boolean local_normalize;
	
	
	private double min;
	private double max;
	
	private int from_x,from_y;
	private int to_x,to_y;

	private int white_level;
	private int black_level;

	private class Coords_listener implements MouseListener, ActionListener
	{
		public void mouseClicked(MouseEvent e)
		{
			Integer a,b;
			Double c;
			

			a= new Integer((int)(e.getX()/panel.get_x_resize_coef()));
			b= new Integer((int)(e.getY()/panel.get_y_resize_coef()));
			c= new Double(mtr.get_element(a.intValue(),b.intValue()));
			
			from.setText(a.toString());
			to.setText(b.toString());
			current_value.setText(c.toString());

			panel.set_pointer_x_coord(a.intValue());
			panel.set_pointer_y_coord(b.intValue());
			panel.repaint();
		}

		public void mouseClicked2(MouseEvent e)
		{
			Integer a,b;
			Double c;
			

			a= new Integer((int)(e.getX()/panel.get_x_resize_coef()));
			b= new Integer((int)(e.getY()/panel.get_y_resize_coef()));
			c= new Double(mtr.get_element(a.intValue(),b.intValue()));
			
			from2.setText(a.toString());
			to2.setText(b.toString());
			current_value2.setText(c.toString());

			panel.set_pointer_x_coord2(a.intValue());
			panel.set_pointer_y_coord2(b.intValue());
			panel.repaint();
		}

		public void mouseExited(MouseEvent e)
		{
			return;
		}

		public void mouseEntered(MouseEvent e)
		{
			return;
		}

		public void mouseReleased(MouseEvent e)
		{
			this.mouseClicked2(e);
		}

		public void mousePressed(MouseEvent e)
		{
			this.mouseClicked(e);
			this.mouseClicked2(e);
		}

		public void actionPerformed(ActionEvent e)
		{
			Integer a,b,a2,b2;

			Double c,c2;
			try
			{
				a = new Integer(from.getText());
				b = new Integer(to.getText());
				a2 = new Integer(from2.getText());
				b2 = new Integer(to2.getText());
			}
			catch(Exception ex)
			{
				a=new Integer(panel.get_pointer_x_coord());
				b=new Integer(panel.get_pointer_y_coord());
				a2=new Integer(panel.get_pointer_x_coord2());
				b2=new Integer(panel.get_pointer_y_coord2());
				from.setText(a.toString());
				to.setText(b.toString());
				from2.setText(a2.toString());
				to2.setText(b2.toString());
				return;
			}
			
			if(
				(a >= 0 && a < mtr.get_size_x())&&
				( b >= 0 && b < mtr.get_size_y())
			  )
			{
				c = new Double(mtr.get_element(a.intValue(),b.intValue()));
				
				current_value.setText(c.toString());

				panel.set_pointer_x_coord(a.intValue());
				panel.set_pointer_y_coord(b.intValue());
				panel.repaint();
			}
			else
			{
				a=new Integer(panel.get_pointer_x_coord());
				b=new Integer(panel.get_pointer_y_coord());
				from.setText(a.toString());
				to.setText(b.toString());
			}

			if(
				(a2 >= 0 && a2 < mtr.get_size_x())&&
				( b2 >= 0 && b2 < mtr.get_size_y())
			  )
			{
				c2 = new Double(mtr.get_element(a2.intValue(),b2.intValue()));
				
				current_value2.setText(c2.toString());

				panel.set_pointer_x_coord2(a2.intValue());
				panel.set_pointer_y_coord2(b2.intValue());
				panel.repaint();
			}
			else
			{
				a2=new Integer(panel.get_pointer_x_coord2());
				b2=new Integer(panel.get_pointer_y_coord2());
				from2.setText(a2.toString());
				to2.setText(b2.toString());
			}
			
		}
	}

	private class Coords_listener2 extends Coords_listener
	{
		public void mouseClicked(MouseEvent e)
		{
			Integer a,b;
			Double c;
			

			a= new Integer((int)(e.getX()/panel.get_x_resize_coef())+from_x);
			b= new Integer((int)(e.getY()/panel.get_y_resize_coef())+from_y);
			c= new Double(mtr.get_element(a.intValue(),b.intValue()));
			
			from.setText(a.toString());
			to.setText(b.toString());
			current_value.setText(c.toString());

			panel.set_pointer_x_coord(a.intValue()-from_x);
			panel.set_pointer_y_coord(b.intValue()-from_y);
			panel.repaint();
		}
		public void mouseClicked2(MouseEvent e)
		{
			Integer a,b;
			Double c;
			

			a= new Integer((int)(e.getX()/panel.get_x_resize_coef())+from_x);
			b= new Integer((int)(e.getY()/panel.get_y_resize_coef())+from_y);
			c= new Double(mtr.get_element(a.intValue(),b.intValue()));
			
			from2.setText(a.toString());
			to2.setText(b.toString());
			current_value2.setText(c.toString());

			panel.set_pointer_x_coord2(a.intValue()-from_x);
			panel.set_pointer_y_coord2(b.intValue()-from_y);
			panel.repaint();
		}		
		public void mouseReleased(MouseEvent e)
		{
			//this.mouseClicked(e);			
			this.mouseClicked2(e);
		}
		public void actionPerformed(ActionEvent e)
		{
			Integer a,b,a2,b2;

			Double c,c2;
			try
			{
				a = new Integer(from.getText());
				b = new Integer(to.getText());
				a2 = new Integer(from2.getText());
				b2 = new Integer(to2.getText());
			}
			catch(Exception ex)
			{
				a=new Integer(panel.get_pointer_x_coord()+from_x);
				b=new Integer(panel.get_pointer_y_coord()+from_y);
				a2=new Integer(panel.get_pointer_x_coord2()+from_x);
				b2=new Integer(panel.get_pointer_y_coord2()+from_y);
				from.setText(a.toString());
				to.setText(b.toString());
				from2.setText(a2.toString());
				to2.setText(b2.toString());
				return;
			}
			
			if(
				(a >= from_x && a <=to_x)&&
				( b >= from_y && b <= to_y)
			  )
			{
				c = new Double(mtr.get_element(a.intValue(),b.intValue()));
				
				current_value.setText(c.toString());

				panel.set_pointer_x_coord(a.intValue()-from_x);
				panel.set_pointer_y_coord(b.intValue()-from_y);
				//panel.set_pointer_x_coord2(a.intValue()-from_x);
				//panel.set_pointer_y_coord2(b.intValue()-from_y);
				
				panel.repaint();
			}
			else
			{
				a=new Integer(panel.get_pointer_x_coord()+from_x);
				b=new Integer(panel.get_pointer_y_coord()+from_y);
				from.setText(a.toString());
				to.setText(b.toString());
			}

			if(
				(a2 >= from_x && a2 <=to_x)&&
				( b2 >= from_y && b2 <= to_y)
			  )
			{	
				
				c2 = new Double(mtr.get_element(a2.intValue(),b2.intValue()));
				
				current_value2.setText(c2.toString());

				panel.set_pointer_x_coord2(a2.intValue());
				panel.set_pointer_y_coord2(b2.intValue());
				panel.repaint();
			}
			else
			{
				a2=new Integer(panel.get_pointer_x_coord2());
				b2=new Integer(panel.get_pointer_y_coord2());
				from2.setText(a2.toString());
				to2.setText(b2.toString());
			}
			
		}
	}
	private class Zoom_action implements ActionListener
	{
		public void actionPerformed(ActionEvent e)
		{
			int fromx=panel.get_pointer_x_coord()+from_x;
			int fromy=panel.get_pointer_y_coord()+from_y;
			
			int tox=panel.get_pointer_x_coord2()+from_x;
			int toy=panel.get_pointer_y_coord2()+from_y;

			
			Matrix_shower shower=new Matrix_shower(tframe,mtr,deviation,true,fromx,fromy,tox,toy);
			shower.setTitle	("Zoom ("+(fromx>=tox?tox:fromx)+","+
						(fromy>=toy?toy:fromy)+"):("+
						(fromx<=tox?tox:fromx)+","+
						(fromy<=toy?toy:fromy)+")"
					);
			shower.setVisible(true);
		}
	}

	private class Slider_listener implements ChangeListener
	{

		private int slider_type;

		public void stateChanged(ChangeEvent e)
		{
			if(slider_type==0)
			{
				white_level=white_slider.getValue();
			}
			
			if(slider_type==1)
			{
				black_level=black_slider.getValue();
			}

			update(mtr,deviation);
		}

		public Slider_listener(int type)
		{
			slider_type=type;
		}
	}

	public Matrix_shower(JDialog frame, Matrix m,Matrix d,boolean normolize)
	{
		super(frame);
		tframe = frame;
		mtr=m;
		deviation=d;
		local_normalize=normolize;
		from_x=0;
		from_y=0;

		white_level=100;
		black_level=100;

		white_slider=new JSlider(javax.swing.SwingConstants.VERTICAL,0,100,100);
		black_slider=new JSlider(javax.swing.SwingConstants.VERTICAL,0,100,100);
		Slider_listener white_listener=new Slider_listener(0 /* white slider*/);
		Slider_listener black_listener=new Slider_listener(1 /* black slider*/);

		white_slider.addChangeListener(white_listener);
		black_slider.addChangeListener(black_listener);


		Coords_listener listener = new Coords_listener();

		Container pane=this.getContentPane();
		
		from = new JTextField("0");
		from.addActionListener(listener);
		
		to   = new JTextField("0");
		to.addActionListener(listener);
	
		from2 = new JTextField("0");
		from2.addActionListener(listener);
		
		to2   = new JTextField("0");
		to2.addActionListener(listener);
		
		Double val= new Double(mtr.get_element(0,0));
		current_value = new JTextField(val.toString());
		current_value2 = new JTextField(val.toString());
	
		panel = new Matrix_panel(mtr,deviation,true);
		panel.setSize(300,300);
		panel.addMouseListener(listener);

		JPanel main_panel=new JPanel();

		main_panel.setLayout(new BorderLayout());
		main_panel.add(BorderLayout.CENTER,panel);
		
		JPanel coords_panel=new JPanel();
		JButton button=new JButton("Change coordinates");
		button.addActionListener(listener);
		JButton button1=new JButton("Zoom");
		button1.addActionListener(new Zoom_action());

		coords_panel.setLayout(new GridLayout(2,4));
		coords_panel.add(from);
		coords_panel.add(to);
		coords_panel.add(current_value);
		coords_panel.add(button);
		coords_panel.add(from2);
		coords_panel.add(to2);
		coords_panel.add(current_value2);
		coords_panel.add(button1);
		
		
		main_panel.add(BorderLayout.SOUTH , coords_panel);
		main_panel.add(BorderLayout.EAST  , black_slider);
		main_panel.add(BorderLayout.WEST  , white_slider);
		
			
		pane.add(main_panel);
		
		/*
		URL path=Objects_loader.get_file("images/logotype.png");
		ImageIcon icon=new ImageIcon(path);
		*/

		this.setSize(340,350);
		/*
		 * Supported from java 1.6
		this.setIconImage(icon.getImage());
		*/
		

	}

	public Matrix_shower(JDialog frame, Matrix m,Matrix d,boolean normolize,int fromx, int fromy, int tox, int toy)
	{
		super(frame);
		tframe = frame;
		local_normalize=true;//normolize;
		mtr=m;
		deviation=d;
		from_x=(fromx>=tox?tox:fromx);
		from_y=fromy>=toy?toy:fromy;
		to_x=fromx<=tox?tox:fromx;
		to_y=fromy<=toy?toy:fromy;


		Coords_listener listener = new Coords_listener2();

		Container pane=this.getContentPane();
		
		from = new JTextField(""+from_x);
		from.addActionListener(listener);
		
		to   = new JTextField(""+from_y);
		to.addActionListener(listener);

		from2 = new JTextField(""+from_x);
		from2.addActionListener(listener);
		
		to2   = new JTextField(""+from_y);
		to2.addActionListener(listener);
		
		Double val= new Double(mtr.get_element(from_x,from_y));
		current_value = new JTextField(val.toString());
		current_value2 = new JTextField(val.toString());
	
		panel = new Matrix_panel(mtr,deviation,true,fromx,fromy,tox,toy);
		panel.setSize(20,20);
		panel.addMouseListener(listener);

		JPanel main_panel=new JPanel();

		main_panel.setLayout(new BorderLayout());
		main_panel.add(BorderLayout.CENTER,panel);
		
		JPanel coords_panel=new JPanel();
		JButton button=new JButton("Change coordinates");
		button.addActionListener(listener);
		JButton button1=new JButton("Zoom");
		button1.addActionListener(new Zoom_action());

		coords_panel.setLayout(new GridLayout(2,4));
		coords_panel.add(from);
		coords_panel.add(to);
		coords_panel.add(current_value);
		coords_panel.add(button);
		coords_panel.add(from2);
		coords_panel.add(to2);
		coords_panel.add(current_value2);
		coords_panel.add(button1);
	
		white_slider=new JSlider(javax.swing.SwingConstants.VERTICAL,0,100,100);
		black_slider=new JSlider(javax.swing.SwingConstants.VERTICAL,0,100,100);
		Slider_listener white_listener=new Slider_listener(0 /* white slider*/);
		Slider_listener black_listener=new Slider_listener(1 /* black slider*/);

		white_slider.addChangeListener(white_listener);
		black_slider.addChangeListener(black_listener);

	
		
		main_panel.add(BorderLayout.SOUTH,coords_panel);
		main_panel.add(BorderLayout.EAST  , black_slider);
		main_panel.add(BorderLayout.WEST  , white_slider);
		
			
		pane.add(main_panel);
		
		/*
		URL path=Objects_loader.get_file("images/logotype.png");
		ImageIcon icon=new ImageIcon(path);
		*/

		this.setSize(300,350);
		/*
		 * Supported from java 1.6
		this.setIconImage(icon.getImage());
		*/
		

	}

	public int get_pointer_x_coord()
	{
		return panel.get_pointer_x_coord();
	}

	public int get_pointer_y_coord()
	{
		return panel.get_pointer_y_coord();
	}

	public int get_pointer_x_coord2()
	{
		return panel.get_pointer_x_coord2();
	}

	public int get_pointer_y_coord2()
	{
		return panel.get_pointer_y_coord2();
	}

	public void update(Matrix m,Matrix d)
	{
		mtr=m;
		deviation=d;
	
		if(local_normalize==true)
		{
		 	double	min=mtr.get_min_element();
			double  max=mtr.get_max_element();

			panel.set_min_matrix_value(min+(max-min)*(100-black_level)/100.0);
			panel.set_max_matrix_value(max-(max-min)*(100-white_level)/100.0);
		}
		panel.update(mtr,deviation);
		
		Double c= new Double(mtr.get_element(panel.get_pointer_x_coord(),panel.get_pointer_y_coord()));
		current_value.setText(c.toString());
		
		panel.repaint();
	}

	public void set_min(double m)
	{
		panel.set_min_matrix_value(m);
	}

	public void set_max(double m)
	{
		panel.set_max_matrix_value(m);
	}

	public double get_min()
	{
		return panel.get_min_matrix_value();
	}

	public double get_max()
	{
		return panel.get_max_matrix_value();
	}

	public void set_local_normalization(boolean norm)
	{
		local_normalize=norm;
		this.update(mtr,deviation);
	}

	public boolean get_local_normalization()
	{
		return local_normalize;
	}

	public static void main(String args[])
	{
		Matrix m=new Matrix(1000,1000);

		for(int i=0;i < 1000; i++)
		for(int j=0; j< 1000; j++)
		{
			m.set_element(i,j,(double)(i+j));
		}
		
		Matrix_shower shower = new Matrix_shower(null,m,null,true);
		shower.setVisible(true);
	}

	

}

